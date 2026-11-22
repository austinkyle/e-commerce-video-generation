from __future__ import annotations

import argparse
import json
import webbrowser
from decimal import Decimal
from pathlib import Path

from ecom_video.config import load_config
from ecom_video.costs import estimate_generation, estimate_within_approved_amount, price_openai_analysis_usage
from ecom_video.media import probe_video, sample_frames
from ecom_video.models import AnalysisResult
from ecom_video.workflow import (
    MultiClipPlan,
    build_multiclip_plan,
    creative_instruction_length,
    estimate_multiclip_generation,
    generation_approval_required,
    run_multiclip_generation,
)


def _inspect(args: argparse.Namespace) -> None:
    info = probe_video(Path(args.video))
    config = load_config()
    frame_dir = Path(config["data_dir"]) / "frames" / Path(info["path"]).stem
    frames = sample_frames(Path(info["path"]), frame_dir, args.frames)
    print(json.dumps({**info, "frames": [str(p) for p in frames]}, indent=2))


def _estimate(args: argparse.Namespace) -> None:
    estimate = estimate_generation(args.seconds, args.images)
    print(f"Estimated image/video generation: ${estimate.image_usd + estimate.video_usd:.4f} USD")
    print(f"Analysis allowance (estimate):     ${estimate.analysis_usd:.4f} USD")
    print(f"Estimated total:                   ${estimate.total_usd:.4f} USD")
    print(f"Video seconds requested/generated: {estimate.requested_seconds}/{estimate.generated_seconds}")
    print("Provider rounding, failures, and retries may change actual charges.")


def _analyze(args: argparse.Namespace) -> None:
    # This command is the user's explicit initiation of paid reference analysis.
    from ecom_video.providers.openai import analyze_reference

    frames = [Path(p) for p in args.frame]
    if creative_instruction_length(args.instruction) > 20_000:
        raise ValueError("creative instructions must be 20,000 characters or fewer")
    if len(frames) > 24:
        raise ValueError("At most 24 frames may be analyzed")
    for path in frames:
        if not path.is_file():
            raise FileNotFoundError(path)
    print(f"Submitting reference analysis to OpenAI (estimated small cost: about ${Decimal('0.02'):.2f}).")
    result = analyze_reference(frames, args.instruction, args.product_facts)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2) + "\n")
    print(f"Analysis saved to {output}")
    usage = analyze_reference.last_usage
    if usage:
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cached_tokens = usage.get("input_tokens_details", {}).get("cached_tokens", 0)
        actual_estimate = price_openai_analysis_usage(usage)
        print(f"OpenAI usage: {input_tokens} input ({cached_tokens} cached), {output_tokens} output tokens")
        if actual_estimate is not None:
            print(f"OpenAI charge estimate from reported usage: ${actual_estimate:.6f}")


def _prepare_multiclip(args: argparse.Namespace) -> None:
    analysis_path = Path(args.analysis)
    if not analysis_path.is_file():
        raise FileNotFoundError(analysis_path)
    analysis = AnalysisResult.model_validate_json(analysis_path.read_text())
    plan = build_multiclip_plan(
        analysis,
        instructions=args.instructions,
        product_facts=args.product_facts,
        environment=args.environment,
        clip_count=args.clip_count,
        seconds_per_clip=args.seconds_per_clip,
        product_references=[Path(path) for path in args.product_reference],
        continuity_references=[Path(path) for path in args.continuity_reference],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(plan.model_dump_json(indent=2) + "\n")
    estimate = estimate_multiclip_generation(plan)
    print(f"Multi-clip plan saved to {output}")
    print(f"Distinct clips: {len(plan.clips)} ({estimate.generated_seconds}s total)")
    print(f"Starting images: ${estimate.image_usd:.4f} USD")
    print(f"Video generation: ${estimate.video_usd:.4f} USD")
    print(f"Generation estimate: ${estimate.total_usd:.4f} USD")
    print("Reference analysis is reused from the provided JSON; no analysis API call was made.")
    print("No image or video generation API calls were made.")


def _generate_multiclip(args: argparse.Namespace) -> None:
    plan_path = Path(args.plan)
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    plan = MultiClipPlan.model_validate_json(plan_path.read_text())
    estimate = estimate_multiclip_generation(plan)
    print(f"Starting images: ${estimate.image_usd:.4f} USD")
    print(f"Video generation: ${estimate.video_usd:.4f} USD")
    print(f"Estimated generation total: ${estimate.total_usd:.4f} USD")
    if args.dry_run:
        print("Dry run only. No paid generation calls made.")
        return
    pricing_verified = not args.require_approval
    estimate_reliable = not args.require_approval
    if args.approve_max_usd is not None and not estimate_within_approved_amount(estimate.total_usd, args.approve_max_usd):
        raise PermissionError(
            f"Estimated cost ${estimate.total_usd:.4f} exceeds approved maximum ${args.approve_max_usd:.4f}"
        )
    approval_required = generation_approval_required(
        estimate,
        pricing_verified=pricing_verified,
        estimate_reliable=estimate_reliable,
    )
    if approval_required and args.approve_max_usd is None:
        print("No paid generation calls made. Confirmation is required; rerun with --approve-max-usd after approval.")
        return

    from ecom_video.providers.kie import KieClient

    client = KieClient()
    try:
        outputs = run_multiclip_generation(
            plan,
            [Path(path) for path in plan.product_references],
            [Path(path) for path in plan.continuity_references],
            Path(args.output_dir),
            client,
            approved_max_usd=Decimal(str(args.approve_max_usd)) if args.approve_max_usd is not None else None,
            pricing_verified=pricing_verified,
            estimate_reliable=estimate_reliable,
        )
    finally:
        client.close()
    print("Generated silent clips:")
    for output in outputs:
        print(f"  {output}")
    usage_path = Path(args.output_dir) / "generation-usage.json"
    if usage_path.is_file():
        usage = json.loads(usage_path.read_text())
        print(f"Usage record: {usage_path}")
        if usage["all_task_charges_reported"]:
            print(
                f"Kie actual usage: {usage['total_credits_consumed']} credits "
                f"(${usage['total_usd_equivalent']} USD equivalent)"
            )
        else:
            print("Kie task usage is incomplete; see the usage record for reported task charges.")


def _serve(args: argparse.Namespace) -> None:
    from ecom_video.web_app import create_server

    server = create_server(args.port)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"E-Commerce Video Generator is running at {url}")
    print("Keep this terminal window open while using the local workspace. Press Ctrl-C to stop.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("Stopping the local workspace.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="ecom-video")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inspect a video and extract local representative frames")
    inspect.add_argument("video")
    inspect.add_argument("--frames", type=int, default=12)
    inspect.set_defaults(run=_inspect)
    estimate = commands.add_parser("estimate", help="Estimate image and video generation costs")
    estimate.add_argument("--seconds", type=int, required=True)
    estimate.add_argument("--images", type=int, default=0)
    estimate.set_defaults(run=_estimate)
    analysis = commands.add_parser("analyze", help="Explicitly submit selected frames for paid analysis")
    analysis.add_argument("--frame", action="append", required=True, help="Frame path; repeat up to 24 times")
    analysis.add_argument("--instruction", required=True)
    analysis.add_argument("--product-facts", default="")
    analysis.add_argument("--output", default="data/analysis.json")
    analysis.set_defaults(run=_analyze)
    prepare = commands.add_parser("prepare-clips", help="Plan separate editable clips from saved reference analysis")
    prepare.add_argument("--analysis", required=True, help="Saved AnalysisResult JSON")
    prepare.add_argument("--instructions", required=True, help="Creative direction for all clips")
    prepare.add_argument("--product-facts", required=True, help="Visible product details to preserve")
    prepare.add_argument("--environment", required=True, help="Intended video environment shared by each starting image and clip")
    prepare.add_argument("--clip-count", type=int, default=3)
    prepare.add_argument("--seconds-per-clip", type=int, default=5)
    prepare.add_argument("--product-reference", action="append", required=True, help="Product reference image; repeat as needed")
    prepare.add_argument("--continuity-reference", action="append", default=[], help="Approved appearance/environment image; repeat as needed")
    prepare.add_argument("--output", default="data/multiclip-plan.json")
    prepare.set_defaults(run=_prepare_multiclip)
    generate = commands.add_parser("generate-clips", help="Generate separate clips from a saved plan")
    generate.add_argument("--plan", required=True)
    generate.add_argument("--output-dir", default="data/outputs/multiclip-run")
    generate.add_argument("--approve-max-usd", type=Decimal, help="Explicit maximum for jobs over $0.50 or with uncertain pricing")
    generate.add_argument("--require-approval", action="store_true", help="Hold for confirmation when pricing or the estimate is uncertain")
    generate.add_argument("--dry-run", action="store_true", help="Show estimate and stop without paid calls")
    generate.set_defaults(run=_generate_multiclip)
    serve = commands.add_parser("serve", help="Open the local browser workspace")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-open", action="store_true", help="Start the local server without opening a browser")
    serve.set_defaults(run=_serve)
    args = parser.parse_args()
    try:
        args.run(args)
    except (RuntimeError, ValueError, FileNotFoundError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
