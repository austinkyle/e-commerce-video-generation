# Operating Principles — E-Commerce Video Generation

This project is an experimental AI creative-production tool.

The primary objective is to generate high-quality, realistic product footage quickly, consistently, and inexpensively.

## 1. User References Are Sufficient for Creative Testing

When the user supplies reference videos, product photographs, screenshots, or descriptions:

- Treat those materials as the authoritative visual references for the requested creative test.
- Do not require independent supplier, manufacturer, or retailer verification before generating prototype footage.
- Do not block progress because a retailer website is inaccessible.
- Do not demand confirmation that provisional product photos exactly match a linked retail listing.
- If references differ, prioritize the user's explicitly selected images and instructions.

Commercial product claims, actual shipped packaging, and real retailer availability must still be represented accurately. Do not invent those facts.

## 2. Default to Progress, Not Repeated Clarification

Proceed using reasonable creative assumptions whenever possible.

Ask questions only when:

- Required inputs are genuinely missing.
- Instructions conflict in a way that materially affects the result.
- An important physical characteristic cannot be established well enough to generate believable footage.
- Follow the generation spending policy in section 4; confirmation is needed only above its threshold or when pricing or the estimate is uncertain.

Do not ask for repeated confirmation of creative choices the user has already made.

## 3. Prioritize Creative Execution

The user is testing the ability to:

- Interpret viral reference videos.
- Replace products, people, and settings.
- Generate realistic starting images.
- Produce convincing individual video clips.
- Maintain physical and visual consistency.
- Minimize cost per usable output.

Optimize for these outcomes, not exhaustive verification or procedural approvals.

## 4. Maintain Necessary Financial and Technical Safeguards

- Show estimated costs for transparency, but do not interrupt ordinary generation with a confirmation dialog.
- When the user explicitly starts a job estimated at $0.50 USD or less and provider pricing and the estimate are verified and reliable, proceed without additional spending approval.
- If the entire estimated job exceeds $0.50 USD, or provider pricing cannot be verified, or the estimate is unreliable, request confirmation before spending.
- Never initiate additional paid generations, retries, or unrequested batches without authorization.
- Protect API credentials and local files.
- Record and report actual charges and relevant quality problems after each job.

Apply this same policy in the CLI, tests, and any future browser interface.

Do not implement monthly spending caps or restrictions.

## 5. Avoid Unnecessary Complexity

- Prefer the simplest solution that achieves the requested result.
- Do not add approval stages without a clear benefit.
- Avoid unnecessary product onboarding, brand systems, or supplier-validation workflows.
- Distinguish experimental creative assets from verified commercial product representations.
- Keep implementation focused on the requested milestone.

**Guiding principle: Execute confidently with the information provided. Ask only when something genuinely prevents successful execution.**

## Creative Director Interview Protocol — Future Original-Concept Workflow

This section specifies a future application workflow for original video concepts. Preserve it as project guidance; do not implement the interview feature until the current reference-to-clips validation work is complete and that milestone is authorized.

### Mission

When the user requests an entirely new video concept, the application must act as an experienced AI commercial director, cinematographer, product photographer, and creative strategist. Transform the user's initial idea into a detailed, physically realistic, generation-ready creative plan. Users should not need to know every detail required for successful AI image and video generation; discover missing details through a natural, intelligent interview.

Primary goal: maximize first-generation quality and consistency while minimizing wasted image and video credits.

### 1. When to activate the interview

#### New original concept — activate

Begin the Creative Director Interview when:

- The user is developing an original video idea from scratch.
- There is no reference video establishing the necessary creative structure.
- The user wants a substantially new creative concept rather than another variation of existing footage.
- Available references leave important creative or physical details unresolved.

#### Existing reference or iteration — do not automatically activate

Do not conduct a full interview when:

- The user provides an existing video and wants a straightforward recreation with specified changes.
- The user wants another version of an existing scene.
- The user wants to change only the actor, product variation, location, camera angle, or another defined element.
- The user wants to regenerate an unsuccessful clip.

For these requests, use known project information and ask only questions essential to execution. Never make users repeat answers already provided.

### 2. Interview methodology

Conduct a conversational, adaptive interview. Do not present a massive static questionnaire. Ask questions in logical groups, generally three to five at a time, and use answers to determine what needs clarification next. Tailor questions to the actual product and concept; explain technical choices in ordinary language.

When the user says “surprise me,” “you decide,” or delegates a creative decision, make an informed recommendation rather than continuing to ask questions. Distinguish between:

- Confirmed product facts.
- User-provided creative preferences.
- Reasonable creative assumptions.
- Important unresolved information.

Do not ask questions simply to make the process appear thorough. Match interview depth to video complexity: a simple product beauty shot may need only a few questions, while a multi-scene demonstration involving actors, animals, packaging, and physical interactions may need a deeper interview. Do not ask about facts already established.

### 3. Essential areas to investigate

#### A. Creative objective

Understand:

- What product, idea, or experience is being featured.
- The central creative concept and what the viewer should feel.
- What makes the video interesting enough to watch.
- Whether the objective is curiosity, humor, surprise, demonstration, emotional appeal, or something else.
- Which visual moment provides the payoff.
- Whether there is specific inspiration or a reference.

Help strengthen weak concepts rather than merely executing them.

#### B. Product identity and physical accuracy

Determine what matters for realism:

- Physical dimensions and approximate scale.
- Shape, color, materials, texture, and finish.
- Actual product functionality, moving parts, and mechanical limitations.
- How humans or animals physically interact with it and what changes during use.
- What must remain unchanged and which visual references are available.

Ask for measurements only when scale materially affects realism. Never invent product specifications or functionality. Use supplied photographs as the visual source of truth for creative tests. Do not require unnecessary retailer or supplier verification.

#### C. Models, people, and animals

When applicable, establish approximate age and general appearance; clothing, hairstyle, and accessories; natural expressions and emotional reactions; body positions and movement; hand placement and object interaction; whether the same person appears across scenes; and relevant animal size, appearance, positioning, and behavior.

Emphasize realistic anatomy, natural movement, and physical continuity. Avoid unnecessary detailed character creation when the subject is incidental.

#### D. Environment and atmosphere

Determine indoor or outdoor location; relevant background objects; time of day; lighting direction and intensity; weather when relevant; realistic environmental imperfections; desired atmosphere; and whether multiple clips share the same location.

For casual ecommerce UGC, favor authentic environments unless the user requests another aesthetic. The setting in generated starting images must match the intended video environment.

#### E. Camera direction and cinematography

For each necessary shot, establish subject framing and distance; camera position and height; camera perspective (first-person POV, handheld, selfie, tripod, etc.); camera movement; subject movement relative to the camera; focus and depth of field; natural or cinematic lighting; intended video style and visual texture; and approximate duration.

Do not default to polished cinematic footage when casual smartphone realism is more appropriate.

#### F. Physical action and motion

This is a critical interview category. Understand exactly what happens during each shot, including:

- The initial physical state.
- The action that begins the movement.
- What the person, animal, and product do.
- How objects physically interact.
- The final physical state.
- What must remain consistent throughout.

For product demonstrations, identify where force is applied, how objects are supported, and which mechanisms actually cause the desired result. Do not simply instruct a video model to “show a dispenser working”; specify the observable action sequence and resulting state, grounded in the product's real capabilities.

If an interaction is too complicated to generate reliably, recommend a simpler shot that communicates the same idea.

#### G. Product packaging and branding

Investigate this category only if packaging appears in the proposed video, such as an unboxing, holding a branded retail box, removing a product from its package, a first-person view of the package, or a product displayed on a shelf.

Determine whether packaging actually exists or is conceptual; product name to display; box dimensions relative to the product; material and finish; color scheme; logos and graphics; exact visible text; how the box opens; how the product fits inside; and how hands interact with it.

When appropriate, propose a simple, believable product name and packaging concept. Keep text, colors, graphics, and proportions consistent across shots. Do not build a full branding system unless separately requested. Do not represent fictional packaging or retailer placement as verified commercial facts.

#### H. Audio

Silent video is the default. Ask about audio only when it affects the concept. Possible choices include silent footage, natural environmental sounds, product interaction sound effects, human reactions, or spoken dialogue. If dialogue is required, clarify the exact words, speaker, tone, and timing. Do not add background music automatically.

#### I. Hook, B-roll, and visual payoff

Help identify which shots are necessary. A concept may include an opening visual hook, product reveal, close-up details, demonstration, human reaction, or transformation/payoff. Do not force every video into this structure; select only the shots needed to communicate the concept effectively.

### 4. Convert the interview into a generation-ready plan

After gathering sufficient information, internally prepare a detailed creative brief specifying:

- Overall concept and intended visual style.
- Confirmed product facts.
- Necessary creative assumptions.
- Required individual clips and target duration of each.
- Starting-image requirements.
- Exact camera position and shot composition.
- Physical action sequences.
- Product and character continuity.
- Background and lighting.
- Audio requirements.
- Known risks for visual or mechanical realism.
- Details that must never change.

For every clip, construct precise starting-image and video-generation prompts. Prioritize observable visual instructions over vague adjectives. “Ultra-realistic” alone is insufficient: describe the actual lighting, surfaces, movements, camera behavior, and physical interactions that create realism. Use negative instructions selectively for known failure risks, such as distorted fingers, duplicated objects, incorrect labels, floating products, or physically impossible transitions.

### 5. Detect problems before spending credits

Before generating images or footage, evaluate the plan for missing critical information; physically impossible actions; ambiguous product operation; inconsistent dimensions; contradictory scene instructions; unnecessary camera or character changes; unrealistic movement within the requested duration; excessively complex individual shots; potential visual continuity failures; and packaging inconsistencies.

Resolve material problems through targeted questions or simple creative adjustments. Do not try to solve every potential generation risk through extra questions. Prefer the simplest realistic execution.

### 6. User approval and generation

Once the interview is complete:

- Present a concise summary of the final creative direction.
- Identify the clips to be generated.
- Provide an estimated image and video generation cost.
- Allow the user to correct the creative direction.
- Apply the standing generation spending policy in section 4 before paid generation; do not require an extra confirmation for a user-initiated, reliable, verified estimate of $0.50 USD or less.
- Generate necessary assets through the existing application pipeline.

Do not add a separate mandatory storyboard approval process. An optional shot-list preview is acceptable for complex original concepts. Starting images should normally flow automatically into video generation after the applicable spending policy is satisfied, unless the user requests image review or a material issue needs resolution. Never automatically purchase additional generations or retries without authorization.

### 7. Learn from completed generations

After a video is generated, allow the user to describe what succeeded or failed. Use that feedback to improve subsequent prompts and creative decisions. Preserve relevant approved references, product facts, and successful prompt strategies within the project. Do not repeatedly interview the user about established facts, and do not assume every successful creative decision is a universal rule.

### 8. Guiding principle

The system owns the responsibility for asking the right questions. The user owns the creative vision and final approval. The application should feel like working with an experienced commercial director who can translate ideas into realistic AI-generated footage: detailed where necessary, fast where possible, and never unnecessarily complicated.

### Implementation boundary

These instructions must ultimately be implemented in the standalone application's original-concept workflow, not merely followed by Codex during development. Preserve them in `AGENTS.md` now, but implement the interview functionality only when the current reference-to-clips validation work is complete and the original-concept milestone is authorized.
