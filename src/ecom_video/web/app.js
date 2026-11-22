(() => {
  const $ = (id) => document.getElementById(id);
  const MAX_INSTRUCTION_LENGTH = 20_000;
  const DEFAULT_REVIEW_IMAGES_BEFORE_VIDEO = true;
  const state = { job: null, pollTimer: null, draftTimer: null, savingDraft: false, draftSavePending: false, draftSavePromise: null, videoFile: null, savedVideo: null, removeReferenceVideo: false, productFiles: [], savedProductAssets: [], pendingAction: null, pendingDeleteProjectId: null, pendingDeleteProjectIds: [], selectedProjectIds: new Set(), historyJobs: [], toastTimer: null };
  const activeStates = new Set(['queued', 'analyzing_reference', 'interviewing', 'refining_plan', 'preparing_images', 'generating_clips', 'processing_outputs', 'regenerating', 'regenerating_starting_image']);
  const titles = {
    draft: 'Draft saved', queued: 'Queued for analysis', analyzing_reference: 'Creative Director working', interviewing: 'Interview in progress', ready: 'Ready to generate',
    preparing_images: 'Preparing starting images', generating_clips: 'Generating clips', processing_outputs: 'Preparing clip files',
    refining_plan: 'Refining creative plan', regenerating: 'Regenerating a clip', regenerating_starting_image: 'Regenerating a starting image',
    awaiting_image_review: 'Review starting images', completed: 'Ready to review', failed: 'Job needs attention', interrupted: 'Job interrupted'
  };

  async function request(url, options = {}) {
    const response = await fetch(url, { cache: 'no-store', ...options });
    const type = response.headers.get('content-type') || '';
    const payload = type.includes('application/json') ? await response.json() : null;
    if (!response.ok) {
      const error = new Error(payload?.error || `Request failed (${response.status}).`);
      error.payload = payload;
      error.status = response.status;
      throw error;
    }
    return payload;
  }

  function toast(message) {
    const el = $('toast');
    el.textContent = message;
    el.classList.add('show');
    clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => el.classList.remove('show'), 3200);
  }

  function setError(message = '') { $('formError').textContent = message; }
  function assetUrl(asset) { return asset ? asset.url : ''; }
  function assetDownloadUrl(asset) {
    const url = assetUrl(asset);
    return url ? `${url}${url.includes('?') ? '&' : '?'}download=1` : '';
  }
  function fmtMoney(value, digits = 4) {
    if (value === undefined || value === null || value === '') return '—';
    const n = Number(value);
    return Number.isFinite(n) ? `$${n.toFixed(digits)}` : '—';
  }
  function setBusy(busy) {
    const locked = busy || Boolean(state.job && state.job.status !== 'draft');
    $('creationMode').disabled = locked;
    $('hookType').disabled = locked;
    $('analyzeButton').disabled = locked;
    $('videoDrop').disabled = locked;
    $('imageDrop').disabled = locked;
    $('removeVideo').disabled = locked;
    $('videoInput').disabled = locked;
    $('productInput').disabled = locked;
    $('instructions').readOnly = locked;
    $('reviewImagesBeforeVideo').disabled = busy || Boolean(state.job && !['draft', 'ready', 'failed', 'interrupted'].includes(state.job.status));
    $('clipCount').disabled = locked;
    $('clipSeconds').disabled = locked;
    $('generateButton').disabled = busy || !state.job || (state.job.status !== 'ready' && !state.job.can_resume);
  }
  function setLocalVideo(file) {
    state.videoFile = file || null;
    $('selectedVideo').hidden = !file;
    $('videoDrop').hidden = Boolean(file);
    if (file) {
      state.savedVideo = null; state.removeReferenceVideo = false;
      $('localVideoPreview').src = URL.createObjectURL(file);
      $('videoFileName').textContent = file.name;
    } else {
      $('localVideoPreview').removeAttribute('src');
      $('localVideoPreview').load();
      $('videoInput').value = '';
    }
  }
  function setSavedVideo(asset) {
    state.savedVideo = asset || null;
    if (!asset) { setLocalVideo(null); return; }
    state.videoFile = null; state.removeReferenceVideo = false;
    $('selectedVideo').hidden = false; $('videoDrop').hidden = true;
    $('localVideoPreview').src = asset.url; $('videoFileName').textContent = asset.filename;
    $('videoInput').value = '';
  }
  function renderImages() {
    const root = $('imagePreviews');
    root.replaceChildren();
    const items = [
      ...state.savedProductAssets.map((asset) => ({ asset, name: asset.filename, src: asset.url })),
      ...state.productFiles.map((file) => ({ file, name: file.name, src: URL.createObjectURL(file) })),
    ];
    items.forEach((item) => {
      const wrap = document.createElement('div');
      wrap.className = 'image-preview-wrap';
      const tile = document.createElement('div');
      tile.className = 'image-preview';
      const image = document.createElement('img');
      image.src = item.src;
      image.alt = item.name;
      const remove = document.createElement('button');
      remove.type = 'button'; remove.setAttribute('aria-label', `Remove ${item.name}`); remove.textContent = '×';
      remove.disabled = Boolean(state.job && state.job.status !== 'draft');
      remove.addEventListener('click', () => {
        if (item.asset) state.savedProductAssets = state.savedProductAssets.filter((asset) => asset.asset_id !== item.asset.asset_id);
        else state.productFiles = state.productFiles.filter((file) => file !== item.file);
        renderImages(); scheduleDraftSave();
      });
      tile.append(image, remove);
      const name = document.createElement('span'); name.className = 'image-preview-name'; name.textContent = item.name;
      wrap.append(tile, name); root.append(wrap);
    });
  }
  function useVideoFiles(files) {
    const file = files?.[0];
    if (!file) return;
    if (!/\.(mp4|mov)$/i.test(file.name)) { setError('Choose an MP4 or MOV reference video.'); return; }
    if (file.size > 200 * 1024 * 1024) { setError('The reference video must be 200 MB or smaller.'); return; }
    setError(''); setLocalVideo(file); scheduleDraftSave(true);
  }
  function useImageFiles(files) {
    const incoming = [...(files || [])];
    const valid = incoming.filter((file) => /\.(jpe?g|png|webp)$/i.test(file.name) && file.size <= 20 * 1024 * 1024);
    if (valid.length !== incoming.length) toast('Use JPEG, PNG, or WebP product photos up to 20 MB each.');
    const combined = [...state.productFiles, ...valid];
    state.productFiles = combined.slice(0, 16);
    if (state.productFiles.length < combined.length) toast('Product photo limit is 16 images.');
    $('productInput').value = '';
    renderImages(); scheduleDraftSave(true);
  }

  function draftFormData() {
    const form = new FormData();
    form.append('mode', $('creationMode').value);
    form.append('hook', $('hookType').value);
    form.append('instructions', $('instructions').value);
    form.append('clip_count', $('clipCount').value);
    form.append('seconds_per_clip', $('clipSeconds').value);
    form.append('review_images_before_video', String($('reviewImagesBeforeVideo').checked));
    form.append('product_asset_ids', JSON.stringify(state.savedProductAssets.map((asset) => asset.asset_id)));
    form.append('remove_reference_video', String(state.removeReferenceVideo));
    if (state.videoFile) form.append('reference_video', state.videoFile, state.videoFile.name);
    state.productFiles.forEach((file) => form.append('product_images', file, file.name));
    return form;
  }

  function scheduleDraftSave(immediate = false) {
    if (state.job && state.job.status !== 'draft') return;
    if (state.savingDraft) state.draftSavePending = true;
    $('jobStatusTitle').textContent = 'Saving draft';
    $('jobStatusMessage').textContent = 'Saving your inputs locally; analysis will wait for your explicit action.';
    clearTimeout(state.draftTimer);
    state.draftTimer = setTimeout(() => saveDraft().catch((error) => toast(`Draft not saved: ${error.message}`)), immediate ? 0 : 650);
  }

  async function saveDraft() {
    if (state.savingDraft) {
      await state.draftSavePromise;
      if (state.draftSavePending) { state.draftSavePending = false; return saveDraft(); }
      return;
    }
    if (state.job && state.job.status !== 'draft') return;
    state.savingDraft = true;
    state.draftSavePromise = Promise.resolve();
    const sentVideo = state.videoFile;
    const sentImages = [...state.productFiles];
    const savePromise = (async () => { try {
      const endpoint = state.job ? `/api/jobs/${state.job.id}/draft` : '/api/drafts';
      const saved = await request(endpoint, { method: 'POST', body: draftFormData() });
      state.job = saved;
      try { localStorage.setItem('ecomVideoCurrentJobId', saved.id); } catch (_) {}
      const videoWasSaved = state.videoFile === sentVideo;
      if (videoWasSaved) state.videoFile = null;
      state.productFiles = state.productFiles.filter((file) => !sentImages.includes(file));
      state.removeReferenceVideo = false;
      const savedVideo = saved.inputs.find((asset) => asset.role === 'reference_video') || null;
      state.savedProductAssets = saved.inputs.filter((asset) => asset.role === 'product_reference');
      if (videoWasSaved) setSavedVideo(savedVideo);
      renderImages();
      $('topbarTitle').textContent = saved.title || 'New creation';
      $('jobStatusTitle').textContent = titles.draft;
      $('jobStatusMessage').textContent = 'Draft saved locally. Analysis starts only when you submit it.';
      setBusy(false);
      await refreshHistory();
    } finally { state.savingDraft = false; state.draftSavePromise = null; } })();
    state.draftSavePromise = savePromise;
    await savePromise;
    if (state.draftSavePending) { state.draftSavePending = false; return saveDraft(); }
  }

  function updateProgress(job) {
    const stages = ['analyzing_reference', 'preparing_images', 'generating_clips', 'processing_outputs', 'completed'];
    const current = stages.indexOf(job.stage);
    const doneThrough = job.status === 'ready' ? 0 : current - 1;
    document.querySelectorAll('.progress-step').forEach((node, index) => {
      node.classList.toggle('active', activeStates.has(job.status) && index === current);
      node.classList.toggle('done', index <= doneThrough || job.status === 'completed');
    });
    const lamp = $('statusLamp');
    lamp.className = `status-lamp${activeStates.has(job.status) ? ' active' : ['completed', 'ready'].includes(job.status) ? ' done' : job.status === 'failed' ? ' error' : ''}`;
    $('jobStatusTitle').textContent = titles[job.status] || 'Job status';
    $('jobStatusMessage').textContent = job.resume_operation?.description || job.error || job.message || '';
    $('providerState').hidden = !job.provider_state;
    $('providerState').textContent = job.provider_state ? `Provider: ${job.provider_state}` : '';
  }

  function updateEstimate(job) {
    const estimate = job.estimate;
    if (!estimate) {
      $('estimateTotal').innerHTML = '—<small>USD</small>';
      $('estimateDescription').textContent = 'Analyze a reference to see the image and video estimate.';
      $('analysisCost').textContent = '~$0.02'; $('imageCost').textContent = '—'; $('videoCost').textContent = '—';
      return;
    }
    $('estimateTotal').innerHTML = `${fmtMoney(estimate.total_usd)}<small>USD</small>`;
    $('estimateDescription').textContent = `${estimate.clip_count} separate clips · ${estimate.total_video_seconds} seconds generated${estimate.intended_edit_seconds ? ` · about ${estimate.intended_edit_seconds}s intended final edit` : ''}`;
    $('analysisCost').textContent = fmtMoney(estimate.analysis_usd, 4);
    $('imageCost').textContent = fmtMoney(estimate.images_usd, 4);
    $('videoCost').textContent = fmtMoney(estimate.video_usd, 4);
    $('generateButton').disabled = job.status !== 'ready' && !job.can_resume;
    if (job.resume_operation) {
      const retryCost = fmtMoney(job.resume_operation.estimated_usd, 2);
      $('estimateTotal').innerHTML = `${retryCost}<small>USD</small>`;
      $('estimateDescription').textContent = `Starting image retry only · saved images for other clips will be reused`;
      $('analysisCost').textContent = '—'; $('imageCost').textContent = retryCost; $('videoCost').textContent = '$0.00';
    }
    $('generateButton').title = job.resume_operation?.description || (job.can_resume ? 'Resume from saved tasks and outputs' : job.estimate_reliable ? 'Start the planned generation' : 'Estimate is uncertain; confirmation will be requested');
    $('generateButton').querySelector('span').textContent = job.resume_operation?.label || (job.can_resume ? 'Resume clips' : 'Generate clips');
  }

  function renderSpending(job) {
    const section = $('projectSpending');
    section.hidden = !job;
    if (!job) { $('spendingRecords').replaceChildren(); return; }
    const spending = job.spending || { recorded_usd_equivalent: '0.000000', unknown_charge_count: 0, records: [], basis: 'No recorded provider usage is available.' };
    $('projectSpendingTotal').innerHTML = `${fmtMoney(spending.recorded_usd_equivalent, 6)}<small>USD equivalent</small>`;
    const unknown = Number(spending.unknown_charge_count || 0);
    const limit = fmtMoney(job.project_spending_limit_usd || '6.00', 2);
    $('projectSpendingNote').textContent = `Per-project ceiling: ${limit}. This is not permission to spend it automatically. ${spending.basis || 'Recorded provider usage equivalents.'}${unknown ? ` ${unknown} operation${unknown === 1 ? '' : 's'} have unknown charges.` : ''}`;
    const records = spending.records || [];
    $('spendingRecordCount').textContent = `${records.length} operation${records.length === 1 ? '' : 's'}`;
    const root = $('spendingRecords'); root.replaceChildren();
    records.forEach((record) => {
      const item = document.createElement('article'); item.className = `spending-record ${record.status === 'unknown_usage' ? 'spending-unknown' : ''}`;
      const heading = document.createElement('div'); heading.className = 'spending-record-heading';
      const operation = document.createElement('strong'); operation.textContent = `${record.provider} · ${record.operation}`;
      const amount = document.createElement('span');
      if (record.usd_equivalent !== null && record.usd_equivalent !== undefined) amount.textContent = fmtMoney(record.usd_equivalent, 4);
      else if (record.estimated_usd_equivalent !== null && record.estimated_usd_equivalent !== undefined) amount.textContent = `Allowance ${fmtMoney(record.estimated_usd_equivalent, 4)} · not a charge`;
      else amount.textContent = 'Amount unknown';
      heading.append(operation, amount);
      const detail = document.createElement('small');
      const time = record.recorded_at ? new Date(record.recorded_at).toLocaleString() : 'Time not recorded (legacy record)';
      const credits = record.credits ? ` · ${record.credits} credits` : '';
      const task = record.task_id ? ` · Task ${record.task_id}` : '';
      detail.textContent = `${time}${credits}${task} · ${record.basis || record.status}`;
      item.append(heading, detail); root.append(item);
    });
  }

  function renderAnalysis(job) {
    $('analysisSection').hidden = !job.analysis;
    if (!job.analysis) return;
    $('analysisSummary').textContent = job.analysis.summary || `${job.analysis.shot_count} reference shots analyzed.`;
    const questions = $('analysisQuestions');
    questions.replaceChildren();
    (job.analysis.questions || []).forEach((text) => {
      const p = document.createElement('p'); p.textContent = text; questions.append(p);
    });
    $('plannedShotCount').textContent = `${job.clips.length} CLIPS`;
    const shots = $('plannedShots'); shots.replaceChildren();
    job.clips.forEach((clip) => {
      const row = document.createElement('article'); row.className = 'planned-shot';
      const number = document.createElement('span'); number.className = 'planned-shot-number'; number.textContent = String(clip.index).padStart(2, '0');
      const copy = document.createElement('div');
      const heading = document.createElement('strong'); heading.textContent = clip.title || clip.action || `Clip ${clip.index}`;
      const detail = document.createElement('p'); detail.textContent = `${clip.action || ''} ${clip.camera ? `· ${clip.camera}` : ''}`.trim();
      copy.append(heading, detail);
      const timing = document.createElement('small');
      const hasSourceSegment = Number.isFinite(clip.source_start_seconds) && Number.isFinite(clip.source_end_seconds);
      const sourceLabel = job.mode === 'original'
        ? 'Original concept · no source footage'
        : hasSourceSegment
        ? `Reference segment: ${Number(clip.source_start_seconds).toFixed(1)}–${Number(clip.source_end_seconds).toFixed(1)}s`
        : 'No exact reference segment';
      const editLabel = Number.isFinite(clip.edit_duration_seconds) ? ` · Intended final edit: ${clip.edit_duration_seconds}s` : '';
      timing.textContent = `${sourceLabel} · Generated clip: ${clip.duration_seconds}s${editLabel}`;
      row.append(number, copy, timing, createAudioControls(job, clip)); shots.append(row);
    });
  }

  function createAudioControls(job, clip) {
    const root = document.createElement('div'); root.className = 'audio-controls';
    const label = document.createElement('label'); label.textContent = `Clip ${clip.index} audio mode`;
    const select = document.createElement('select'); select.setAttribute('aria-label', `Clip ${clip.index} audio mode`);
    [['auto', 'Auto · Creative Director'], ['enabled', 'Enabled · Request native audio'], ['silent', 'Silent · No audio']].forEach(([value, text]) => {
      const option = document.createElement('option'); option.value = value; option.textContent = text; select.append(option);
    });
    select.value = clip.audio_mode || 'auto'; select.disabled = activeStates.has(job.status);
    label.append(select);
    const direction = document.createElement('small'); direction.className = 'audio-direction'; direction.setAttribute('aria-label', 'Audio direction');
    const selection = clip.audio_mode === 'enabled' ? 'Enabled · native audio requested' : clip.audio_mode === 'silent' ? 'Silent · no native audio' : clip.audio_enabled ? 'Auto · Creative Director selected native audio' : 'Auto · Creative Director selected silent output';
    direction.textContent = `${selection}${clip.audio_direction ? ` — ${clip.audio_direction}` : ''}`;
    const note = document.createElement('small'); note.textContent = 'Applies to the next generation or regeneration.';
    select.addEventListener('change', async () => {
      select.disabled = true;
      try {
        const updated = await request(`/api/jobs/${job.id}/clips/${clip.index}/audio`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode: select.value }) });
        renderJob(updated); toast('Clip audio setting saved.');
      } catch (error) { toast(error.message); renderJob(job); }
    });
    root.append(label, direction, note);
    return root;
  }

  function renderResults(job) {
    const hasAny = job.clips?.some((clip) => clip.versions?.length || clip.status !== 'planned');
    $('resultsSection').hidden = !hasAny;
    if (!hasAny) return;
    const root = $('clipResults'); root.replaceChildren();
    let completed = 0;
    job.clips.forEach((clip) => {
      const versions = clip.versions || [];
      if (versions.length) completed++;
      const selected = versions.at(-1);
      const card = document.createElement('article'); card.className = 'clip-card';
      const header = document.createElement('header'); header.className = 'clip-card-header';
      const title = document.createElement('div');
      const h3 = document.createElement('h3'); h3.textContent = `Clip ${String(clip.index).padStart(2, '0')} · ${clip.title || 'Planned shot'}`;
      const desc = document.createElement('p'); desc.textContent = selected ? `${selected.duration_seconds}s · ${versions.length} version${versions.length === 1 ? '' : 's'}` : (clip.status || 'Waiting to generate').replaceAll('_', ' ');
      title.append(h3, desc); header.append(title);
      const tag = document.createElement('span'); tag.className = 'version-tag'; tag.textContent = selected ? `V${selected.version}` : 'PLANNED'; header.append(tag);
      card.append(header, createAudioControls(job, clip));
      if (selected) {
        const video = document.createElement('video'); video.className = 'clip-video'; video.controls = true; video.playsInline = true; video.preload = 'metadata'; video.src = assetUrl(selected.video);
        const assets = document.createElement('div'); assets.className = 'clip-assets';
        const preview = document.createElement('div'); preview.className = 'starting-preview';
        const image = document.createElement('img'); image.src = assetUrl(selected.starting_image); image.alt = `Clip ${clip.index} starting image`;
        const previewCopy = document.createElement('div');
        const previewTitle = document.createElement('strong'); previewTitle.textContent = 'Starting image';
        const previewSub = document.createElement('small'); previewSub.textContent = selected.revision ? 'Revised version' : 'Used for this clip'; previewCopy.append(previewTitle, previewSub);
        const imageDownload = document.createElement('a'); imageDownload.className = 'asset-download'; imageDownload.href = assetDownloadUrl(selected.starting_image); imageDownload.download = selected.starting_image?.filename || `clip-${clip.index}-starting.png`; imageDownload.textContent = 'Download image';
        preview.append(image, previewCopy, imageDownload); assets.append(preview);
        const actions = document.createElement('div'); actions.className = 'clip-actions';
        const videoDownload = document.createElement('a'); videoDownload.className = 'button button-secondary'; videoDownload.href = assetDownloadUrl(selected.video); videoDownload.download = selected.video?.filename || `clip-${clip.index}.mp4`; videoDownload.textContent = 'Download video';
        const regen = document.createElement('button'); regen.type = 'button'; regen.className = 'button button-secondary'; regen.textContent = 'Regenerate'; regen.disabled = activeStates.has(job.status); regen.addEventListener('click', () => form.hidden = !form.hidden);
        actions.append(videoDownload, regen); assets.append(actions);
        const form = document.createElement('form'); form.className = 'regen-form'; form.hidden = true;
        const input = document.createElement('textarea'); input.required = true; input.maxLength = 1200; input.placeholder = 'Describe the change for this clip only…';
        const submit = document.createElement('button'); submit.type = 'submit'; submit.className = 'button button-primary'; submit.textContent = 'Generate revised clip';
        form.append(input, submit);
        form.addEventListener('submit', async (event) => {
          event.preventDefault();
          try {
            submit.disabled = true;
            await performPaidAction(`/api/jobs/${job.id}/clips/${clip.index}/regenerate`, { instructions: input.value }, 'Regeneration');
          } catch (error) { toast(error.message); submit.disabled = false; }
        });
        assets.append(form);
        if (versions.length > 1) {
          const picker = document.createElement('select'); picker.className = 'version-picker'; picker.setAttribute('aria-label', `Clip ${clip.index} version history`);
          versions.forEach((version) => { const option = document.createElement('option'); option.value = String(version.version); option.textContent = `Version ${version.version}${version.revision ? ` · ${version.revision}` : ''}`; picker.append(option); });
          picker.value = String(selected.version);
          picker.addEventListener('change', () => { const v = versions.find((item) => item.version === Number(picker.value)); video.src = assetUrl(v.video); image.src = assetUrl(v.starting_image); videoDownload.href = assetDownloadUrl(v.video); videoDownload.download = v.video.filename; imageDownload.href = assetDownloadUrl(v.starting_image); imageDownload.download = v.starting_image.filename; previewSub.textContent = v.revision || 'Used for this clip'; tag.textContent = `V${v.version}`; });
          assets.append(picker);
        }
        card.append(video, assets);
      } else {
        const waiting = document.createElement('p'); waiting.className = 'clip-waiting'; waiting.textContent = clip.action || 'This clip will appear here when generation begins.'; card.append(waiting);
      }
      root.append(card);
    });
    $('completedCount').textContent = `${completed} / ${job.clips.length} READY`;
  }

  function renderImageReview(job) {
    const section = $('imageReviewSection');
    const hasReviewImages = Boolean(job.clips?.some((clip) => clip.current_starting_image));
    const reviewable = ['awaiting_image_review', 'regenerating_starting_image', 'failed', 'interrupted'].includes(job.status) && !job.images_approved;
    section.hidden = !hasReviewImages || !reviewable;
    const root = $('startingImageReview'); root.replaceChildren();
    if (!hasReviewImages || !reviewable) return;
    const controlsEnabled = job.status === 'awaiting_image_review';
    const imageChecks = [];
    let approve;
    const syncApproval = () => {
      if (approve) approve.disabled = !controlsEnabled || imageChecks.length !== (job.clips || []).length || imageChecks.some((isReady) => !isReady());
    };
    (job.clips || []).forEach((clip) => {
      const card = document.createElement('article'); card.className = 'starting-image-card';
      if (clip.current_starting_image) {
        const image = document.createElement('img'); image.src = assetUrl(clip.current_starting_image); image.alt = `Clip ${clip.index} starting image: ${clip.title || 'planned shot'}`; card.append(image);
        const isReady = () => image.complete && image.naturalWidth > 0;
        imageChecks.push(isReady);
        image.addEventListener('load', syncApproval);
        image.addEventListener('error', syncApproval);
      } else {
        const missing = document.createElement('div'); missing.className = 'starting-image-missing'; missing.textContent = 'Starting image will appear when preparation resumes.'; card.append(missing);
      }
      const copy = document.createElement('div'); copy.className = 'starting-image-copy';
      const title = document.createElement('h3'); title.textContent = `Clip ${String(clip.index).padStart(2, '0')} · ${clip.creative_role || clip.title || 'Planned shot'}`;
      const action = document.createElement('p'); action.textContent = `${clip.title && clip.title !== clip.creative_role ? `${clip.title}. ` : ''}${clip.action || 'Review the planned starting frame.'}`;
      const editLabel = Number.isFinite(clip.edit_duration_seconds) ? ` · ${clip.edit_duration_seconds}s intended edit` : '';
      const imageState = document.createElement('small'); imageState.textContent = `Image ${clip.starting_image_history?.length || 1} · ${clip.duration_seconds}s generated${editLabel}`;
      copy.append(title, action, imageState);
      const edit = document.createElement('div'); edit.className = 'starting-image-edit';
      const input = document.createElement('textarea'); input.rows = 2; input.maxLength = 1200; input.placeholder = 'Optional image adjustment, such as tool angle or hand placement…'; input.setAttribute('aria-label', `Image adjustment for clip ${clip.index}`);
      const button = document.createElement('button'); button.type = 'button'; button.className = 'button button-secondary'; button.textContent = 'Regenerate Image · ~$0.03'; button.disabled = !controlsEnabled;
      button.addEventListener('click', async () => {
        try { await performPaidAction(`/api/jobs/${job.id}/clips/${clip.index}/starting-image/regenerate`, { instructions: input.value.trim() }, `Clip ${clip.index} image regeneration`); }
        catch (error) { toast(error.message); }
      });
      edit.append(input, button); card.append(copy, edit); root.append(card);
      const masked = document.createElement('details'); masked.className = 'masked-edit-option';
      const summary = document.createElement('summary'); summary.textContent = 'Optional: preserve a scene with masked editing';
      const help = document.createElement('p'); help.textContent = 'Choose a suitable saved scene image as the base. Character identity references follow the saved character groups, preferring approved anchors; provisional images are not approved automatically. The exact-source saved mask is reused when available; otherwise upload a same-size transparent PNG mask. Transparent pixels are editable and opaque pixels protect the background.';
      const base = document.createElement('select'); base.setAttribute('aria-label', `Scene base for clip ${clip.index}`);
      const placeholder = document.createElement('option'); placeholder.value = ''; placeholder.textContent = 'Choose a scene image'; base.append(placeholder);
      const choices = job.scene_base_images || [];
      for (const choice of choices) {
        const option = document.createElement('option'); option.value = choice.value;
        option.textContent = `${choice.label}${choice.mask_available ? ' · saved mask' : ' · mask upload needed'}`;
        base.append(option);
      }
      const preview = document.createElement('img'); preview.alt = 'Selected scene base preview'; preview.hidden = true;
      const maskFile = document.createElement('input'); maskFile.type = 'file'; maskFile.accept = 'image/png';
      maskFile.setAttribute('aria-label', `Optional localized alpha mask for clip ${clip.index}`);
      const maskStatus = document.createElement('small'); maskStatus.textContent = 'No scene selected. The normal regeneration option remains available above.';
      base.addEventListener('change', () => {
        const choice = choices.find((item) => item.value === base.value);
        preview.hidden = !choice;
        if (choice) preview.src = choice.url;
        maskFile.value = '';
        maskStatus.textContent = !choice ? 'No scene selected.' : choice.mask_available
          ? 'A saved mask for this exact source image will be reused. Upload a replacement mask only if needed.'
          : 'No saved mask matches this source. Upload a same-size transparent PNG to continue.';
        maskedButton.disabled = !controlsEnabled || !choice;
      });
      const maskedButton = document.createElement('button'); maskedButton.type = 'button'; maskedButton.className = 'button button-secondary';
      maskedButton.textContent = 'Generate Masked Candidate · ~$0.05'; maskedButton.disabled = !controlsEnabled || choices.length === 0;
      maskedButton.addEventListener('click', async () => {
        const choice = choices.find((item) => item.value === base.value);
        if (!choice) { toast('Choose an appropriate scene image first.'); return; }
        const file = maskFile.files?.[0];
        if (!choice.mask_available && !file) { toast('Upload a transparent PNG mask for this scene image.'); return; }
        const form = new FormData(); form.set('edit_mode', 'masked'); form.set('base_source', choice.value);
        form.set('instructions', input.value.trim());
        if (file) form.set('mask_image', file, file.name);
        try { await performPaidAction(`/api/jobs/${job.id}/clips/${clip.index}/starting-image/regenerate`, form, `Clip ${clip.index} masked image edit`); }
        catch (error) { toast(error.message); }
      });
      masked.append(summary, help, base, preview, maskFile, maskStatus, maskedButton);
      card.append(masked);
    });
    approve = $('approveImagesButton');
    approve.hidden = job.status !== 'awaiting_image_review';
    approve.textContent = job.clip_regeneration_pending ? 'Approve revised image & generate clip' : 'Approve images & generate videos';
    syncApproval();
  }

  function updateCreationModeUi() {
    const original = $('creationMode').value === 'original';
    $('referenceHeading').closest('.input-panel').hidden = original;
    $('hookField').hidden = false;
    $('analyzeButton').textContent = original ? 'Develop original concept' : 'Analyze reference';
    $('instructions').placeholder = original ? 'Describe your original idea and product. Include anything you already know; the Creative Director will ask what it needs…' : 'Use this video’s creative structure, but feature my product…';
  }

  function restoreSelectValue(select, value, label) {
    const saved = String(value);
    if (![...select.options].some((option) => option.value === saved)) {
      const option = document.createElement('option'); option.value = saved; option.textContent = label; select.append(option);
    }
    select.value = saved;
  }

  function renderJob(job) {
    state.job = job;
    try { localStorage.setItem('ecomVideoCurrentJobId', job.id); } catch (_) {}
    restoreSelectValue($('creationMode'), job.mode || 'reference', job.mode || 'Reference Recreation');
    restoreSelectValue($('hookType'), job.hook || 'recommend', job.hook || 'Recommend a hook');
    const countValue = job.clip_count_selection === 'director' ? 'director' : String(job.requested_clip_count || 3);
    const durationValue = job.duration_selection === 'director' ? 'director' : String(job.seconds_per_clip || 5);
    restoreSelectValue($('clipCount'), countValue, countValue === 'director' ? 'Director decides' : `${countValue} clips`);
    restoreSelectValue($('clipSeconds'), durationValue, durationValue === 'director' ? 'Director decides' : `${durationValue} sec`);
    $('instructions').value = job.instructions || '';
    $('characterCount').textContent = `${$('instructions').value.length.toLocaleString()} / ${MAX_INSTRUCTION_LENGTH.toLocaleString()}`;
    $('reviewImagesBeforeVideo').checked = Boolean(job.review_images_before_video || job.status === 'awaiting_image_review');
    const reference = (job.inputs || []).find((asset) => asset.role === 'reference_video');
    state.productFiles = [];
    state.savedProductAssets = (job.inputs || []).filter((asset) => asset.role === 'product_reference');
    setSavedVideo(reference || null);
    renderImages();
    updateCreationModeUi();
    $('topbarTitle').textContent = job.title || 'Current creation';
    $('currentProjectStorage').textContent = `${formatBytes(job.storage_bytes || 0)} on disk`;
    $('currentProjectStorage').hidden = false;
    $('deleteCurrentProject').hidden = false;
    $('analysisHeading').textContent = job.mode === 'original' ? 'Creative Plan' : 'Reference analysis';
    updateProgress(job); updateEstimate(job); renderSpending(job); renderAnalysis(job); renderResults(job); renderImageReview(job);
    const interview = $('interviewSection');
    interview.hidden = !(job.mode === 'original' && job.status === 'interviewing');
    $('directorQuestions').replaceChildren(...(job.director_questions || []).map((question) => { const p = document.createElement('p'); p.textContent = question; return p; }));
    if (job.status === 'interviewing') $('analysisSection').hidden = true;
    $('refinePlanForm').hidden = !(job.status === 'ready' && job.clips?.length);
    $('updatePlanButton').disabled = activeStates.has(job.status);
    $('generateButton').disabled = (job.status !== 'ready' && !job.can_resume) || job.status === 'awaiting_image_review';
    setBusy(activeStates.has(job.status));
    if (activeStates.has(job.status)) pollLater(job.id);
  }

  function pollLater(jobId, delay = 1600) {
    clearTimeout(state.pollTimer);
    state.pollTimer = setTimeout(async () => {
      try {
        const job = await request(`/api/jobs/${jobId}`);
        renderJob(job);
        if (!activeStates.has(job.status)) await refreshHistory();
      } catch (error) { $('jobStatusMessage').textContent = error.message; }
    }, delay);
  }

  async function refreshHistory() {
    try {
      const response = await request('/api/jobs');
      state.historyJobs = response.jobs;
      state.selectedProjectIds = new Set([...state.selectedProjectIds].filter((id) => response.jobs.some((job) => job.id === id)));
      $('storageSummary').textContent = `Managed storage: ${formatBytes(response.storage?.total_bytes || 0)} · ${response.storage?.project_count || 0} projects`;
      const root = $('historyList'); root.replaceChildren();
      if (!response.jobs.length) { const p = document.createElement('p'); p.className = 'history-empty'; p.textContent = 'Your recent projects will appear here.'; root.append(p); updateBulkControls(); return; }
      response.jobs.forEach((job) => {
        const name = job.title || 'Project';
        const row = document.createElement('div'); row.className = 'history-row';
        const select = document.createElement('input'); select.type = 'checkbox'; select.className = 'history-select'; select.checked = state.selectedProjectIds.has(job.id); select.disabled = activeStates.has(job.status); select.setAttribute('aria-label', `Select ${name}`);
        select.addEventListener('change', () => { select.checked ? state.selectedProjectIds.add(job.id) : state.selectedProjectIds.delete(job.id); updateBulkControls(); });
        const button = document.createElement('button'); button.className = `history-item${state.job?.id === job.id ? ' selected' : ''}`; button.type = 'button';
        button.title = name;
        const nameLabel = document.createElement('span'); nameLabel.className = 'history-name'; nameLabel.textContent = name;
        const meta = document.createElement('small'); meta.textContent = `${titles[job.status] || job.status} · ${formatBytes(job.storage_bytes || 0)} · ${new Date(job.updated_at).toLocaleDateString()}`;
        button.append(nameLabel, meta); button.addEventListener('click', async () => { try { renderJob(await request(`/api/jobs/${job.id}`)); await refreshHistory(); } catch (error) { toast(error.message); } });
        const edit = document.createElement('button'); edit.className = 'history-rename'; edit.type = 'button';
        edit.setAttribute('aria-label', `Rename ${name}`); edit.title = `Rename ${name}`;
        const icon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        icon.setAttribute('viewBox', '0 0 24 24'); icon.setAttribute('aria-hidden', 'true');
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', 'M12 20h9 M16.5 3.5a2.1 2.1 0 0 1 3 3L9 17l-4 1 1-4Z');
        icon.append(path); edit.append(icon);
        edit.addEventListener('click', (event) => {
          event.stopPropagation();
          const input = document.createElement('input'); input.className = 'history-name-input';
          input.type = 'text'; input.value = name; input.maxLength = 100;
          input.setAttribute('aria-label', `Project name, currently ${name}`);
          nameLabel.replaceWith(input); edit.hidden = true; input.focus(); input.select();
          let settling = false;
          const cancel = () => {
            if (settling) return;
            settling = true; input.replaceWith(nameLabel); edit.hidden = false;
          };
          const save = async () => {
            if (settling) return;
            const title = input.value.trim();
            input.setCustomValidity(title ? '' : 'Project name cannot be empty.');
            if (!title) { input.reportValidity(); input.focus(); return; }
            if (title === name) { cancel(); return; }
            settling = true; input.disabled = true;
            try {
              const updated = await request(`/api/jobs/${job.id}/rename`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title }),
              });
              if (state.job?.id === job.id) {
                state.job = updated;
                $('topbarTitle').textContent = updated.title;
              }
              await refreshHistory(); toast('Project renamed.');
            } catch (error) {
              settling = false; input.disabled = false; input.focus(); toast(error.message);
            }
          };
          input.addEventListener('keydown', (keyEvent) => {
            if (keyEvent.key === 'Enter') { keyEvent.preventDefault(); save(); }
            if (keyEvent.key === 'Escape') { keyEvent.preventDefault(); cancel(); }
          });
        });
        const remove = document.createElement('button'); remove.className = 'history-delete'; remove.type = 'button';
        remove.setAttribute('aria-label', `Delete ${name}`); remove.title = `Delete ${name}`;
        const trash = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        trash.setAttribute('viewBox', '0 0 24 24'); trash.setAttribute('aria-hidden', 'true');
        const trashPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        trashPath.setAttribute('d', 'M4 7h16 M10 11v6 M14 11v6 M5 7l1 13h12l1-13 M9 7V4h6v3');
        trash.append(trashPath); remove.append(trash);
        remove.addEventListener('click', (event) => {
          event.stopPropagation();
          state.pendingDeleteProjectId = job.id;
          openDeleteConfirmation([job.id], name);
        });
        row.append(select, button, edit, remove); root.append(row);
      });
      updateBulkControls();
      if (!state.job) {
        let currentId = null;
        try { currentId = localStorage.getItem('ecomVideoCurrentJobId'); } catch (_) {}
        if (currentId === 'new') return;
        if (!currentId) currentId = response.jobs[0]?.id || null;
        if (currentId && response.jobs.some((job) => job.id === currentId)) {
          renderJob(await request(`/api/jobs/${currentId}`));
          await refreshHistory();
        } else if (currentId) {
          try { localStorage.removeItem('ecomVideoCurrentJobId'); } catch (_) {}
        }
      }
    } catch (_) { /* The workspace remains usable when history cannot be read. */ }
  }

  function formatBytes(value) {
    const bytes = Math.max(0, Number(value) || 0);
    if (bytes < 1024) return `${bytes} B`;
    const units = ['KB', 'MB', 'GB', 'TB']; let size = bytes / 1024; let unit = units[0];
    for (let i = 1; size >= 1024 && i < units.length; i++) { size /= 1024; unit = units[i]; }
    return `${size.toFixed(size >= 100 ? 0 : 1)} ${unit}`;
  }

  function updateBulkControls() {
    const count = state.selectedProjectIds.size;
    $('deleteSelectedProjects').disabled = count === 0;
    $('deleteSelectedProjects').textContent = count ? `Delete selected (${count})` : 'Delete selected';
  }

  async function openDeleteConfirmation(ids, label) {
    try {
      const estimate = await request('/api/jobs/delete-estimate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ job_ids: ids }) });
      state.pendingDeleteProjectIds = ids;
      state.pendingDeleteProjectId = ids.length === 1 ? ids[0] : null;
      $('deleteProjectName').textContent = label;
      $('deleteEstimate').textContent = `${estimate.project_count} project${estimate.project_count === 1 ? '' : 's'} · approximately ${formatBytes(estimate.recoverable_bytes)} recoverable`;
      $('confirmDeleteProject').textContent = ids.length === 1 ? 'Delete Project' : `Delete ${estimate.project_count} Projects`;
      $('confirmDeleteProject').disabled = false;
      $('deleteProjectDialog').showModal();
    } catch (error) { toast(error.message); }
  }

  async function performPaidAction(url, body, label) {
    const submit = async (approved) => {
      if (body instanceof FormData) {
        const form = new FormData(); for (const [key, value] of body.entries()) form.append(key, value);
        if (approved !== undefined) form.set('approved_max_usd', approved);
        return request(url, { method: 'POST', body: form });
      }
      return request(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...body, ...(approved !== undefined ? { approved_max_usd: approved } : {}) }) });
    };
    try {
      renderJob(await submit()); toast(`${label} started.`); await refreshHistory();
    } catch (error) {
      if (!error.payload?.confirmation_required) throw error;
      $('approvalCopy').textContent = error.message;
      $('approvalEstimate').textContent = fmtMoney(error.payload.estimate_usd);
      state.pendingAction = async () => {
        try {
          $('confirmApproval').disabled = true;
          renderJob(await submit(error.payload.estimate_usd));
          $('approvalDialog').close(); state.pendingAction = null; toast(`${label} started.`); await refreshHistory();
        } finally { $('confirmApproval').disabled = false; }
      };
      $('approvalDialog').showModal();
    }
  }

  $('videoDrop').addEventListener('click', () => $('videoInput').click());
  $('videoInput').addEventListener('change', (event) => useVideoFiles(event.target.files));
  $('removeVideo').addEventListener('click', () => { state.removeReferenceVideo = Boolean(state.savedVideo || state.videoFile); setLocalVideo(null); state.savedVideo = null; scheduleDraftSave(true); });
  $('imageDrop').addEventListener('click', () => $('productInput').click());
  $('productInput').addEventListener('change', (event) => useImageFiles(event.target.files));
  [['videoDrop', useVideoFiles], ['imageDrop', useImageFiles]].forEach(([id, handler]) => {
    const zone = $(id);
    zone.addEventListener('dragover', (event) => { event.preventDefault(); zone.classList.add('dragging'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragging'));
    zone.addEventListener('drop', (event) => { event.preventDefault(); zone.classList.remove('dragging'); handler(event.dataTransfer.files); });
  });
  $('instructions').addEventListener('input', () => { $('characterCount').textContent = `${$('instructions').value.length.toLocaleString()} / ${MAX_INSTRUCTION_LENGTH.toLocaleString()}`; scheduleDraftSave(); });
  $('creationMode').addEventListener('change', () => { updateCreationModeUi(); scheduleDraftSave(); });
  ['hookType', 'clipCount', 'clipSeconds'].forEach((id) => $(id).addEventListener('change', () => scheduleDraftSave()));
  $('reviewImagesBeforeVideo').addEventListener('change', async () => {
    if (!state.job) return;
    const checkbox = $('reviewImagesBeforeVideo');
    const enabled = checkbox.checked;
    if (state.job.status === 'draft') { scheduleDraftSave(); return; }
    checkbox.disabled = true;
    try {
      renderJob(await request(`/api/jobs/${state.job.id}/review-images`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled })
      }));
      await refreshHistory();
    } catch (error) {
      checkbox.checked = Boolean(state.job.review_images_before_video);
      toast(error.message);
    } finally {
      setBusy(activeStates.has(state.job?.status));
    }
  });
  $('creationForm').addEventListener('submit', async (event) => {
    event.preventDefault(); setError('');
    const original = $('creationMode').value === 'original';
    if (!original && !state.videoFile && !state.savedVideo) { setError('Choose a reference video to continue.'); $('videoDrop').focus(); return; }
    if (!$('instructions').value.trim()) { setError('Add creative instructions before analyzing.'); $('instructions').focus(); return; }
    if ($('instructions').value.length > MAX_INSTRUCTION_LENGTH) { setError('Keep creative instructions to 20,000 characters or fewer.'); return; }
    try {
      setBusy(true); $('jobStatusTitle').textContent = 'Saving project'; $('jobStatusMessage').textContent = 'Saving your inputs before analysis.';
      if (!state.job || state.job.status === 'draft') await saveDraft();
      renderJob(await request(`/api/jobs/${state.job.id}/analyze`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }));
      await refreshHistory();
    } catch (error) { setBusy(false); setError(error.message); $('jobStatusTitle').textContent = 'Waiting for a reference'; $('jobStatusMessage').textContent = 'Choose a video to begin.'; }
  });
  $('answerForm').addEventListener('submit', async (event) => {
    event.preventDefault(); if (!state.job) return;
    const answer = $('interviewAnswer').value.trim(); if (!answer) return;
    try { $('interviewAnswer').disabled = true; renderJob(await request(`/api/jobs/${state.job.id}/answer`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ answer }) })); $('interviewAnswer').value = ''; await refreshHistory(); }
    catch (error) { toast(error.message); }
    finally { $('interviewAnswer').disabled = false; }
  });
  $('refinePlanForm').addEventListener('submit', async (event) => {
    event.preventDefault(); if (!state.job) return;
    const instructions = $('refinePlanInput').value.trim(); if (!instructions) return;
    try {
      $('updatePlanButton').disabled = true;
      renderJob(await request(`/api/jobs/${state.job.id}/refine`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ instructions }) }));
      $('refinePlanInput').value = ''; toast('Creative plan updated. Review the clips and estimate.'); await refreshHistory();
    } catch (error) { toast(error.message); $('updatePlanButton').disabled = false; }
  });
  $('generateButton').addEventListener('click', async () => {
    if (!state.job) return;
    const resume = Boolean(state.job.can_resume);
    const body = { resume };
    if (!resume) body.review_images = $('reviewImagesBeforeVideo').checked;
    try { await performPaidAction(`/api/jobs/${state.job.id}/generate`, body, resume ? 'Resume' : 'Generation'); }
    catch (error) { toast(error.message); }
  });
  $('approveImagesButton').addEventListener('click', async () => {
    if (!state.job) return;
    const button = $('approveImagesButton'); button.disabled = true;
    try { await performPaidAction(`/api/jobs/${state.job.id}/approve-images`, {}, 'Video generation'); }
    catch (error) { toast(error.message); }
    finally { button.disabled = false; }
  });
  $('confirmApproval').addEventListener('click', async () => { if (state.pendingAction) { try { await state.pendingAction(); } catch (error) { toast(error.message); } } });
  $('cancelApproval').addEventListener('click', () => { state.pendingAction = null; $('approvalDialog').close(); });
  $('cancelDeleteProject').addEventListener('click', () => $('deleteProjectDialog').close());
  $('deleteProjectDialog').addEventListener('close', () => { state.pendingDeleteProjectId = null; state.pendingDeleteProjectIds = []; $('confirmDeleteProject').disabled = false; $('confirmDeleteProject').textContent = 'Delete Project'; });
  $('confirmDeleteProject').addEventListener('click', async () => {
    const projectIds = state.pendingDeleteProjectIds;
    if (!projectIds.length) return;
    const wasOpen = projectIds.includes(state.job?.id);
    const confirmButton = $('confirmDeleteProject');
    confirmButton.disabled = true;
    try {
      if (projectIds.length === 1) await request(`/api/jobs/${projectIds[0]}/delete`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
      else await request('/api/jobs/delete-bulk', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ job_ids: projectIds }) });
      projectIds.forEach((id) => state.selectedProjectIds.delete(id));
      $('deleteProjectDialog').close();
      if (wasOpen) $('newProject').click();
      await refreshHistory();
      toast(projectIds.length === 1 ? 'Project deleted.' : `${projectIds.length} projects deleted.`);
    } catch (error) {
      confirmButton.disabled = false;
      toast(error.message);
    }
  });
  $('selectCompletedProjects').addEventListener('click', () => {
    state.historyJobs.filter((job) => job.status === 'completed').forEach((job) => state.selectedProjectIds.add(job.id));
    refreshHistory();
  });
  $('deleteSelectedProjects').addEventListener('click', () => {
    const ids = [...state.selectedProjectIds];
    if (ids.length) openDeleteConfirmation(ids, `${ids.length} selected projects`);
  });
  $('deleteCurrentProject').addEventListener('click', () => {
    if (state.job) openDeleteConfirmation([state.job.id], state.job.title || 'Project');
  });
  $('newProject').addEventListener('click', () => {
    clearTimeout(state.pollTimer); state.job = null; state.pendingAction = null;
    try { localStorage.setItem('ecomVideoCurrentJobId', 'new'); } catch (_) {}
    state.savedVideo = null; state.savedProductAssets = []; state.removeReferenceVideo = false;
    $('creationForm').reset(); $('characterCount').textContent = `0 / ${MAX_INSTRUCTION_LENGTH.toLocaleString()}`; setLocalVideo(null); state.productFiles = []; renderImages();
    $('reviewImagesBeforeVideo').checked = DEFAULT_REVIEW_IMAGES_BEFORE_VIDEO;
    $('analysisSection').hidden = true; $('imageReviewSection').hidden = true; $('startingImageReview').replaceChildren(); $('resultsSection').hidden = true; $('clipResults').replaceChildren(); setError('');
    $('topbarTitle').textContent = 'New creation'; $('currentProjectStorage').hidden = true; $('deleteCurrentProject').hidden = true; $('jobStatusTitle').textContent = 'Waiting for a reference'; $('jobStatusMessage').textContent = 'Choose a video to begin.';
    $('statusLamp').className = 'status-lamp'; $('generateButton').disabled = true; updateEstimate({}); renderSpending(null); updateCreationModeUi(); window.scrollTo({ top: 0, behavior: 'smooth' });
    setBusy(false);
  });

  updateEstimate({}); refreshHistory();
})();
