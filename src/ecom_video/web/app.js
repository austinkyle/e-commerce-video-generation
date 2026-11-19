(() => {
  const $ = (id) => document.getElementById(id);
  const MAX_INSTRUCTION_LENGTH = 20_000;
  const state = { job: null, pollTimer: null, videoFile: null, productFiles: [], pendingAction: null, pendingDeleteProjectId: null, toastTimer: null };
  const activeStates = new Set(['queued', 'analyzing_reference', 'interviewing', 'refining_plan', 'preparing_images', 'generating_clips', 'processing_outputs', 'regenerating', 'regenerating_starting_image']);
  const titles = {
    queued: 'Queued for analysis', analyzing_reference: 'Creative Director working', interviewing: 'Interview in progress', ready: 'Ready to generate',
    preparing_images: 'Preparing starting images', generating_clips: 'Generating clips', processing_outputs: 'Preparing silent files',
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
    const locked = busy || Boolean(state.job);
    $('analyzeButton').disabled = locked;
    $('videoDrop').disabled = locked;
    $('imageDrop').disabled = locked;
    $('removeVideo').disabled = locked;
    $('videoInput').disabled = locked;
    $('productInput').disabled = locked;
    $('instructions').readOnly = locked;
    $('reviewImagesBeforeVideo').disabled = busy || Boolean(state.job && !['ready', 'failed', 'interrupted'].includes(state.job.status));
    $('clipCount').disabled = locked;
    $('clipSeconds').disabled = locked;
    $('generateButton').disabled = busy || !state.job || (state.job.status !== 'ready' && !state.job.can_resume);
  }
  function setLocalVideo(file) {
    state.videoFile = file || null;
    $('selectedVideo').hidden = !file;
    $('videoDrop').hidden = Boolean(file);
    if (file) {
      $('localVideoPreview').src = URL.createObjectURL(file);
      $('videoFileName').textContent = file.name;
    } else {
      $('localVideoPreview').removeAttribute('src');
      $('localVideoPreview').load();
      $('videoInput').value = '';
    }
  }
  function renderImages() {
    const root = $('imagePreviews');
    root.replaceChildren();
    state.productFiles.forEach((file, index) => {
      const wrap = document.createElement('div');
      wrap.className = 'image-preview-wrap';
      const tile = document.createElement('div');
      tile.className = 'image-preview';
      const image = document.createElement('img');
      image.src = URL.createObjectURL(file);
      image.alt = file.name;
      const remove = document.createElement('button');
      remove.type = 'button'; remove.setAttribute('aria-label', `Remove ${file.name}`); remove.textContent = '×';
      remove.addEventListener('click', () => { state.productFiles.splice(index, 1); renderImages(); });
      tile.append(image, remove);
      const name = document.createElement('span'); name.className = 'image-preview-name'; name.textContent = file.name;
      wrap.append(tile, name); root.append(wrap);
    });
  }
  function useVideoFiles(files) {
    const file = files?.[0];
    if (!file) return;
    if (!/\.(mp4|mov)$/i.test(file.name)) { setError('Choose an MP4 or MOV reference video.'); return; }
    if (file.size > 200 * 1024 * 1024) { setError('The reference video must be 200 MB or smaller.'); return; }
    setError(''); setLocalVideo(file);
  }
  function useImageFiles(files) {
    const incoming = [...(files || [])];
    const valid = incoming.filter((file) => /\.(jpe?g|png|webp)$/i.test(file.name) && file.size <= 20 * 1024 * 1024);
    if (valid.length !== incoming.length) toast('Use JPEG, PNG, or WebP product photos up to 20 MB each.');
    const combined = [...state.productFiles, ...valid];
    state.productFiles = combined.slice(0, 16);
    if (state.productFiles.length < combined.length) toast('Product photo limit is 16 images.');
    $('productInput').value = '';
    renderImages();
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
    $('jobStatusMessage').textContent = job.error || job.message || '';
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
    $('estimateDescription').textContent = `${estimate.clip_count} separate ${estimate.seconds_per_clip}-second clips · ${estimate.total_video_seconds} seconds generated${estimate.intended_edit_seconds ? ` · about ${estimate.intended_edit_seconds}s intended final edit` : ''}`;
    $('analysisCost').textContent = fmtMoney(estimate.analysis_usd, 4);
    $('imageCost').textContent = fmtMoney(estimate.images_usd, 4);
    $('videoCost').textContent = fmtMoney(estimate.video_usd, 4);
    $('generateButton').disabled = job.status !== 'ready' && !job.can_resume;
    $('generateButton').title = job.can_resume ? 'Resume from saved tasks and outputs' : job.estimate_reliable ? 'Start the planned generation' : 'Estimate is uncertain; confirmation will be requested';
    $('generateButton').querySelector('span').textContent = job.can_resume ? 'Resume clips' : 'Generate clips';
  }

  function renderSpending(job) {
    const section = $('projectSpending');
    section.hidden = !job;
    if (!job) { $('spendingRecords').replaceChildren(); return; }
    const spending = job.spending || { recorded_usd_equivalent: '0.000000', unknown_charge_count: 0, records: [], basis: 'No recorded provider usage is available.' };
    $('projectSpendingTotal').innerHTML = `${fmtMoney(spending.recorded_usd_equivalent, 6)}<small>USD equivalent</small>`;
    const unknown = Number(spending.unknown_charge_count || 0);
    $('projectSpendingNote').textContent = `${spending.basis || 'Recorded provider usage equivalents.'}${unknown ? ` ${unknown} operation${unknown === 1 ? '' : 's'} have unknown charges.` : ''}`;
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
      timing.textContent = `${sourceLabel} · Target clip: ${clip.duration_seconds} seconds`;
      row.append(number, copy, timing); shots.append(row);
    });
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
      card.append(header);
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
      const imageState = document.createElement('small'); imageState.textContent = `Image ${clip.starting_image_history?.length || 1} · ${clip.duration_seconds}s planned clip`;
      copy.append(title, action, imageState);
      const edit = document.createElement('div'); edit.className = 'starting-image-edit';
      const input = document.createElement('textarea'); input.rows = 2; input.maxLength = 1200; input.placeholder = 'Optional image adjustment, such as tool angle or hand placement…'; input.setAttribute('aria-label', `Image adjustment for clip ${clip.index}`);
      const button = document.createElement('button'); button.type = 'button'; button.className = 'button button-secondary'; button.textContent = 'Regenerate Image · ~$0.03'; button.disabled = !controlsEnabled;
      button.addEventListener('click', async () => {
        try { await performPaidAction(`/api/jobs/${job.id}/clips/${clip.index}/starting-image/regenerate`, { instructions: input.value.trim() }, `Clip ${clip.index} image regeneration`); }
        catch (error) { toast(error.message); }
      });
      edit.append(input, button); card.append(copy, edit); root.append(card);
    });
    approve = $('approveImagesButton');
    approve.hidden = job.status !== 'awaiting_image_review';
    approve.textContent = job.clip_regeneration_pending ? 'Approve revised image & generate clip' : 'Approve images & generate videos';
    syncApproval();
  }

  function renderJob(job) {
    state.job = job;
    $('reviewImagesBeforeVideo').checked = Boolean(job.review_images_before_video || job.status === 'awaiting_image_review');
    $('topbarTitle').textContent = job.title || 'Current creation';
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
      const root = $('historyList'); root.replaceChildren();
      if (!response.jobs.length) { const p = document.createElement('p'); p.className = 'history-empty'; p.textContent = 'Your recent projects will appear here.'; root.append(p); return; }
      response.jobs.forEach((job) => {
        const name = job.title || 'Project';
        const row = document.createElement('div'); row.className = 'history-row';
        const button = document.createElement('button'); button.className = `history-item${state.job?.id === job.id ? ' selected' : ''}`; button.type = 'button';
        button.title = name;
        const nameLabel = document.createElement('span'); nameLabel.className = 'history-name'; nameLabel.textContent = name;
        const meta = document.createElement('small'); meta.textContent = `${titles[job.status] || job.status} · ${new Date(job.updated_at).toLocaleDateString()}`;
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
          $('deleteProjectName').textContent = name;
          $('confirmDeleteProject').disabled = false;
          $('deleteProjectDialog').showModal();
        });
        row.append(button, edit, remove); root.append(row);
      });
    } catch (_) { /* The workspace remains usable when history cannot be read. */ }
  }

  async function performPaidAction(url, body, label) {
    const submit = async (approved) => request(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...body, ...(approved !== undefined ? { approved_max_usd: approved } : {}) }) });
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
  $('removeVideo').addEventListener('click', () => setLocalVideo(null));
  $('imageDrop').addEventListener('click', () => $('productInput').click());
  $('productInput').addEventListener('change', (event) => useImageFiles(event.target.files));
  [['videoDrop', useVideoFiles], ['imageDrop', useImageFiles]].forEach(([id, handler]) => {
    const zone = $(id);
    zone.addEventListener('dragover', (event) => { event.preventDefault(); zone.classList.add('dragging'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragging'));
    zone.addEventListener('drop', (event) => { event.preventDefault(); zone.classList.remove('dragging'); handler(event.dataTransfer.files); });
  });
  $('instructions').addEventListener('input', () => { $('characterCount').textContent = `${$('instructions').value.length.toLocaleString()} / ${MAX_INSTRUCTION_LENGTH.toLocaleString()}`; });
  $('creationMode').addEventListener('change', () => {
    const original = $('creationMode').value === 'original';
    $('referenceHeading').closest('.input-panel').hidden = original;
    $('hookField').hidden = !original;
    $('analyzeButton').textContent = original ? 'Develop original concept' : 'Analyze reference';
    $('instructions').placeholder = original ? 'Describe your original idea and product. Include anything you already know; the Creative Director will ask what it needs…' : 'Use this video’s creative structure, but feature my product…';
  });
  $('creationForm').addEventListener('submit', async (event) => {
    event.preventDefault(); setError('');
    const original = $('creationMode').value === 'original';
    if (!original && !state.videoFile) { setError('Choose a reference video to continue.'); $('videoDrop').focus(); return; }
    if (!$('instructions').value.trim()) { setError('Add creative instructions before analyzing.'); $('instructions').focus(); return; }
    if ($('instructions').value.length > MAX_INSTRUCTION_LENGTH) { setError('Keep creative instructions to 20,000 characters or fewer.'); return; }
    const form = new FormData();
    if (state.videoFile) form.append('reference_video', state.videoFile, state.videoFile.name);
    state.productFiles.forEach((file) => form.append('product_images', file, file.name));
    form.append('instructions', $('instructions').value);
    form.append('mode', original ? 'original' : 'reference'); form.append('hook', $('hookType').value);
    form.append('clip_count', $('clipCount').value); form.append('seconds_per_clip', $('clipSeconds').value);
    form.append('review_images_before_video', String($('reviewImagesBeforeVideo').checked));
    try {
      setBusy(true); $('jobStatusTitle').textContent = original ? 'Starting interview' : 'Submitting analysis'; $('jobStatusMessage').textContent = original ? 'The Creative Director is reviewing your idea.' : 'The selected reference is being prepared for analysis.';
      renderJob(await request('/api/jobs', { method: 'POST', body: form }));
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
    try { renderJob(await request(`/api/jobs/${state.job.id}/approve-images`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })); toast('Images approved. Video generation started.'); await refreshHistory(); }
    catch (error) { toast(error.message); button.disabled = false; }
  });
  $('confirmApproval').addEventListener('click', async () => { if (state.pendingAction) { try { await state.pendingAction(); } catch (error) { toast(error.message); } } });
  $('cancelApproval').addEventListener('click', () => { state.pendingAction = null; $('approvalDialog').close(); });
  $('cancelDeleteProject').addEventListener('click', () => $('deleteProjectDialog').close());
  $('deleteProjectDialog').addEventListener('close', () => { state.pendingDeleteProjectId = null; $('confirmDeleteProject').disabled = false; });
  $('confirmDeleteProject').addEventListener('click', async () => {
    const projectId = state.pendingDeleteProjectId;
    if (!projectId) return;
    const wasOpen = state.job?.id === projectId;
    const confirmButton = $('confirmDeleteProject');
    confirmButton.disabled = true;
    try {
      await request(`/api/jobs/${projectId}/delete`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
      $('deleteProjectDialog').close();
      if (wasOpen) $('newProject').click();
      await refreshHistory();
      toast('Project deleted.');
    } catch (error) {
      confirmButton.disabled = false;
      toast(error.message);
    }
  });
  $('newProject').addEventListener('click', () => {
    clearTimeout(state.pollTimer); state.job = null; state.pendingAction = null;
    $('creationForm').reset(); $('characterCount').textContent = `0 / ${MAX_INSTRUCTION_LENGTH.toLocaleString()}`; setLocalVideo(null); state.productFiles = []; renderImages();
    $('reviewImagesBeforeVideo').checked = false;
    $('analysisSection').hidden = true; $('imageReviewSection').hidden = true; $('startingImageReview').replaceChildren(); $('resultsSection').hidden = true; $('clipResults').replaceChildren(); setError('');
    $('topbarTitle').textContent = 'New creation'; $('jobStatusTitle').textContent = 'Waiting for a reference'; $('jobStatusMessage').textContent = 'Choose a video to begin.';
    $('statusLamp').className = 'status-lamp'; $('generateButton').disabled = true; updateEstimate({}); renderSpending(null); window.scrollTo({ top: 0, behavior: 'smooth' });
    setBusy(false);
  });

  updateEstimate({}); refreshHistory();
})();
