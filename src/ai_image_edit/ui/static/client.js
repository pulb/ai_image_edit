window.AIU = window.AIU || {};
window.AIE = window.AIE || {};
window.AIS = window.AIS || {};

// Shared helpers.
// A widget's state object: the pan/zoom fields plus whatever it adds.
AIU.newPanZoomState = function (extra) {
    return Object.assign({ scale: 1, panX: 0, panY: 0 }, extra);
};

AIU.applyPanZoom = function (wrap, state) {
    wrap.style.transform = 'translate(' + state.panX + 'px, ' + state.panY + 'px) scale(' + state.scale + ')';
};

AIU.resetPanZoom = function (wrap, state) {
    state.scale = 1;
    state.panX = 0;
    state.panY = 0;
    if (wrap) AIU.applyPanZoom(wrap, state);
};

// Pan/zoom shared by the editor and the comparison slider: two-finger pinch
// or Ctrl/Cmd + mouse wheel (which is also what a trackpad pinch sends)
// zooms wrap (clamped 1x-8x), and once zoomed in a single finger or a mouse
// drag pans. A plain wheel is left alone so the page still scrolls.
// state holds scale/panX/panY; enabled() gates the gestures.
AIU.attachPanZoom = function (wrap, state, enabled) {
    function distance(touches) {
        return Math.hypot(touches[0].clientX - touches[1].clientX, touches[0].clientY - touches[1].clientY);
    }
    function clampPan() {
        // offsetWidth/offsetHeight are the element's own layout size, which
        // a transform on that same element never affects — using
        // getBoundingClientRect() here instead would read back (possibly
        // stale) post-transform geometry, since this runs before the new
        // scale has actually been applied to the DOM.
        const maxX = wrap.offsetWidth * (state.scale - 1) / 2;
        const maxY = wrap.offsetHeight * (state.scale - 1) / 2;
        state.panX = Math.min(maxX, Math.max(-maxX, state.panX));
        state.panY = Math.min(maxY, Math.max(-maxY, state.panY));
    }

    let pinchStartDist = null;
    let pinchStartScale = 1;
    let panStart = null;

    wrap.addEventListener('touchstart', function (e) {
        if (!enabled()) return;
        if (e.touches.length === 2) {
            pinchStartDist = distance(e.touches);
            pinchStartScale = state.scale;
            panStart = null;
        } else if (e.touches.length === 1 && state.scale > 1) {
            panStart = { x: state.panX, y: state.panY, tx: e.touches[0].clientX, ty: e.touches[0].clientY };
        }
    }, { passive: true });

    wrap.addEventListener('touchmove', function (e) {
        if (!enabled()) return;
        if (e.touches.length === 2 && pinchStartDist) {
            state.scale = Math.min(Math.max(pinchStartScale * distance(e.touches) / pinchStartDist, 1), 8);
        } else if (e.touches.length === 1 && panStart) {
            state.panX = panStart.x + (e.touches[0].clientX - panStart.tx);
            state.panY = panStart.y + (e.touches[0].clientY - panStart.ty);
        } else {
            return;
        }
        e.preventDefault();
        clampPan();
        AIU.applyPanZoom(wrap, state);
    }, { passive: false });

    wrap.addEventListener('touchend', function (e) {
        if (e.touches.length < 2) pinchStartDist = null;
        if (e.touches.length < 1) panStart = null;
    });

    wrap.addEventListener('wheel', function (e) {
        if (!enabled() || !(e.ctrlKey || e.metaKey)) return;
        e.preventDefault();
        // Zoom about the cursor: scaling happens about the element's centre,
        // which the translation moves by pan, so the cursor is measured from
        // the untransformed centre.
        const rect = wrap.getBoundingClientRect();
        const cx = e.clientX - (rect.left + rect.width / 2) + state.panX;
        const cy = e.clientY - (rect.top + rect.height / 2) + state.panY;
        const next = Math.min(Math.max(state.scale * Math.exp(-e.deltaY * 0.002), 1), 8);
        const ratio = next / state.scale;
        state.panX = cx - (cx - state.panX) * ratio;
        state.panY = cy - (cy - state.panY) * ratio;
        state.scale = next;
        clampPan();
        AIU.applyPanZoom(wrap, state);
    }, { passive: false });

    let dragStart = null;
    wrap.addEventListener('mousedown', function (e) {
        if (!enabled() || e.button !== 0 || state.scale <= 1) return;
        e.preventDefault();
        dragStart = { x: state.panX, y: state.panY, mx: e.clientX, my: e.clientY };
    });
    window.addEventListener('mousemove', function (e) {
        if (!dragStart) return;
        state.panX = dragStart.x + (e.clientX - dragStart.mx);
        state.panY = dragStart.y + (e.clientY - dragStart.my);
        clampPan();
        AIU.applyPanZoom(wrap, state);
    });
    window.addEventListener('mouseup', function () { dragStart = null; });
};

AIE.init = function (id) {
    const canvas = document.getElementById(id + '_canvas');
    const maskCanvas = document.createElement('canvas');
    const overlayCanvas = document.getElementById(id + '_overlay');
    const annotCanvas = document.getElementById(id + '_annot');

    // brush is stored in on-screen CSS pixels (what the slider and preview
    // circle actually show) — getScale() converts it to canvas coordinate
    // space at draw time, since the canvas's internal resolution (the
    // image's native size) is usually far larger than its rendered size.
    // mode is 'zoom' (the default), 'mask' or 'annotate'; the radio group
    // in create_mask_editor switches it. They are mutually exclusive, so a
    // single finger can safely mean "pan" in zoom mode, since drawing is
    // only ever active in the other two. Annotations keep a fixed thin
    // stroke (annotStroke, CSS pixels) in the selected colour.
    const state = AIU.newPanZoomState({
        canvas: canvas, ctx: canvas.getContext('2d'),
        maskCanvas: maskCanvas, overlayCanvas: overlayCanvas, annotCanvas: annotCanvas,
        zoomwrap: document.getElementById(id + '_zoomwrap'),
        img: null, drawing: false, brush: __DEFAULT_BRUSH_SIZE__, lastX: 0, lastY: 0, hasMask: false,
        feather: __DEFAULT_FEATHER_AMOUNT__, hasAnnotation: false, annotColor: '__ANNOTATION_COLOR__', annotStroke: __ANNOTATION_STROKE__,
        mode: 'zoom',
    });
    // A mask stroke is painted onto two layers: the exported mask, opaque
    // white, and the on-screen overlay in the tint colour, opaque on the
    // canvas and shown at 50% by CSS (.aie-overlay), so repeated strokes
    // don't build up. An annotation stroke goes onto the single opaque
    // annotation layer, which is exported on its own (transparent
    // background) and composed onto the source image by the server.
    const maskLayers = [
        [maskCanvas.getContext('2d'), '#ffffff'],
        [overlayCanvas.getContext('2d'), '__MASK_TINT__'],
    ];
    const annotCtx = annotCanvas.getContext('2d');

    // The layers and width (canvas pixels) the current mode draws with.
    function brushSpec() {
        if (state.mode === 'annotate') {
            return { layers: [[annotCtx, state.annotColor]], width: state.annotStroke * getScale() };
        }
        return { layers: maskLayers, width: state.brush * getScale() };
    }
    AIE[id] = state;
    AIE.showPlaceholder(id, true);

    function getScale() {
        const rect = canvas.getBoundingClientRect();
        return rect.width ? canvas.width / rect.width : 1;
    }

    // canvas.getBoundingClientRect() already reflects the zoomwrap's
    // current CSS transform (scale/translate), so drawing coordinates stay
    // correct at any zoom/pan level without any changes here.
    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const t = e.touches && e.touches.length ? e.touches[0] : e;
        return {
            x: (t.clientX - rect.left) * canvas.width / rect.width,
            y: (t.clientY - rect.top) * canvas.height / rect.height,
        };
    }

    function dot(x, y) {
        const spec = brushSpec();
        const r = spec.width / 2;
        for (const [c, color] of spec.layers) {
            c.beginPath(); c.fillStyle = color;
            c.arc(x, y, r, 0, Math.PI * 2); c.fill();
        }
    }

    function lineTo(x0, y0, x1, y1) {
        const spec = brushSpec();
        for (const [c, color] of spec.layers) {
            c.strokeStyle = color; c.lineWidth = spec.width;
            c.lineCap = 'round'; c.lineJoin = 'round';
            c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke();
        }
    }

    function down(e) {
        if (state.mode === 'zoom' || !state.img) return;
        if (e.touches && e.touches.length !== 1) return;
        e.preventDefault();
        state.drawing = true;
        const p = getPos(e);
        state.lastX = p.x; state.lastY = p.y;
        dot(p.x, p.y);
        if (state.mode === 'annotate') {
            state.hasAnnotation = true;
        } else if (!state.hasMask) {
            state.hasMask = true;
            emitEvent('aie_mask_state', id, true);
        }
    }
    function move(e) {
        if (state.mode === 'zoom' || !state.drawing) return;
        if (e.touches && e.touches.length !== 1) return;
        e.preventDefault();
        const p = getPos(e);
        lineTo(state.lastX, state.lastY, p.x, p.y);
        state.lastX = p.x; state.lastY = p.y;
    }
    function up() { state.drawing = false; }

    canvas.addEventListener('mousedown', down);
    canvas.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
    canvas.addEventListener('touchstart', down, { passive: false });
    canvas.addEventListener('touchmove', move, { passive: false });
    canvas.addEventListener('touchend', up);

    if (state.zoomwrap) AIU.attachPanZoom(state.zoomwrap, state, function () { return state.mode === 'zoom'; });
};

// Empties both mask layers and reports the mask as gone if it wasn't already.
AIE.resetMask = function (id, state) {
    for (const c of [state.maskCanvas, state.overlayCanvas]) {
        c.getContext('2d').clearRect(0, 0, c.width, c.height);
    }
    if (state.hasMask) {
        state.hasMask = false;
        emitEvent('aie_mask_state', id, false);
    }
};

AIE.resetAnnotations = function (state) {
    state.annotCanvas.getContext('2d').clearRect(0, 0, state.annotCanvas.width, state.annotCanvas.height);
    state.hasAnnotation = false;
};

AIE.showPlaceholder = function (id, visible) {
    const ph = document.getElementById(id + '_placeholder');
    if (ph) ph.style.display = visible ? 'flex' : 'none';
    const box = ph && ph.closest('.aie-editor-box');
    if (box) box.classList.toggle('aie-pick', visible);
};

AIE.setMode = function (id, mode) {
    // Brush/Feather sliders are NiceGUI ui.slider widgets, so their enabled
    // state is toggled from Python (see set_mode in create_mask_editor)
    // rather than by reaching into the DOM here.
    const state = AIE[id];
    if (!state) return;
    state.mode = mode;
    state.drawing = false;
};

AIE.resetZoom = function (id) {
    const state = AIE[id];
    if (state) AIU.resetPanZoom(state.zoomwrap, state);
};

AIE.loadImage = function (id, url) {
    const state = AIE[id];
    if (!state) return;
    const img = new Image();
    img.onload = function () {
        state.img = img;
        for (const c of [state.canvas, state.maskCanvas, state.overlayCanvas, state.annotCanvas]) {
            c.width = img.naturalWidth;
            c.height = img.naturalHeight;
        }
        state.ctx.drawImage(img, 0, 0);
        AIE.resetMask(id, state);
        AIE.resetAnnotations(state);
        AIE.resetZoom(id);
        AIE.showPlaceholder(id, false);
    };
    img.src = url;
};

AIE.clearMask = function (id) {
    const state = AIE[id];
    if (state && state.img) AIE.resetMask(id, state);
};

AIE.clearAnnotations = function (id) {
    const state = AIE[id];
    if (state && state.img) AIE.resetAnnotations(state);
};

AIE.clearImage = function (id) {
    const state = AIE[id];
    if (!state) return;
    state.img = null;
    state.drawing = false;
    state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
    AIE.resetMask(id, state);
    AIE.resetAnnotations(state);
    AIE.resetZoom(id);
    AIE.showPlaceholder(id, true);
};

AIE.setBrush = function (id, size) {
    if (AIE[id]) AIE[id].brush = Number(size);
};

AIE.setAnnotationColor = function (id, color) {
    if (AIE[id]) AIE[id].annotColor = color;
};

AIE.showBrushPreview = function (id, size) {
    const el = document.getElementById(id + '_brushpreview');
    if (!el) return;
    el.style.width = size + 'px';
    el.style.height = size + 'px';
    el.style.display = 'block';
};

AIE.hideBrushPreview = function (id) {
    const el = document.getElementById(id + '_brushpreview');
    if (el) el.style.display = 'none';
};

AIE.setFeather = function (id, value) {
    if (AIE[id]) AIE[id].feather = Number(value);
};

// Everything the server needs at Generate time, in one call: the mask and
// the annotation layer (the annotations alone, transparent where nothing was
// drawn) as PNG data URLs, null when not drawn, and the Feather amount.
AIE.getInputs = function (id) {
    const state = AIE[id];
    if (!state || !state.img) return null;
    return {
        mask: state.hasMask ? state.maskCanvas.toDataURL('image/png') : null,
        annotations: state.hasAnnotation ? state.annotCanvas.toDataURL('image/png') : null,
        feather: state.feather,
    };
};

AIS.init = function (id) {
    // sliderValue is tracked here (rather than read back from the slider's
    // own DOM) since range_slider is a Quasar-rendered ui.slider, not a raw
    // <input> with a plain .value property.
    const state = AIU.newPanZoomState({
        wrap: document.getElementById(id + '_zoomwrap'),
        compare: false, sliderValue: 50,
    });
    AIS[id] = state;
    if (state.wrap) AIU.attachPanZoom(state.wrap, state, function () { return true; });
};

AIS.applyClip = function (id, val) {
    const state = AIS[id];
    if (!state) return;
    state.sliderValue = val;
    const before = document.getElementById(id + '_before');
    const handle = document.getElementById(id + '_handle');
    if (before) before.style.clipPath = 'inset(0 ' + (100 - val) + '% 0 0)';
    if (handle) handle.style.left = val + '%';
};

AIS.setImages = function (id, beforeUrl, afterUrl) {
    const state = AIS[id];
    if (!state) return;
    const after = document.getElementById(id + '_after');
    const before = document.getElementById(id + '_before');
    if (after) { after.src = afterUrl; after.style.display = 'block'; }
    if (before) { before.src = beforeUrl; before.style.display = 'block'; }
    // The slider's own value is reset from Python (see set_images in
    // create_compare_slider) — this just re-applies the clip/handle
    // position to match, since a Python-side value push alone doesn't
    // touch the clip-path.
    AIS.applyClip(id, 50);
    AIU.resetPanZoom(state.wrap, state);
    AIS.setCompare(id, state.compare);
};

AIS.setCompare = function (id, enabled) {
    const state = AIS[id];
    if (!state) return;
    state.compare = enabled;
    const range = document.getElementById(id + '_range');
    const handle = document.getElementById(id + '_handle');
    const before = document.getElementById(id + '_before');
    if (!handle || !before) return;
    // Hiding "before" with display:none rather than a clip at 100%: its box
    // is computed independently from "after"'s (object-fit: cover vs. the
    // zoomwrap's auto height), so a clip boundary at exactly 100% can leave
    // a sub-pixel sliver of it visible at the bottom/right edge.
    const display = enabled ? 'block' : 'none';
    before.style.display = display;
    handle.style.display = display;
    if (range) range.style.display = display;
    if (enabled) AIS.applyClip(id, state.sliderValue);
};

AIS.getAfterUrl = function (id) {
    const el = document.getElementById(id + '_after');
    return (el && el.getAttribute('src')) ? el.src : null;
};

AIS.reset = function (id) {
    const state = AIS[id];
    const after = document.getElementById(id + '_after');
    const before = document.getElementById(id + '_before');
    // removeAttribute('src') alone can leave a visible broken-image icon in
    // some browsers once an <img> has previously held a real src — hiding
    // both elements outright avoids that regardless of browser quirks.
    if (after) { after.removeAttribute('src'); after.style.display = 'none'; }
    if (before) { before.removeAttribute('src'); before.style.display = 'none'; }
    if (state) AIU.resetPanZoom(state.wrap, state);
    AIS.setCompare(id, false);
};
