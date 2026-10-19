/* Verification page: sliders, tabs, camera */
(function () {
  /* --- Range sliders --- */
  var thresholdRange = document.getElementById("id_threshold_range");
  var thresholdHidden = document.getElementById("id_threshold");
  var thresholdVal = document.getElementById("threshold-val");
  var brightnessRange = document.getElementById("id_brightness_range");
  var brightnessHidden = document.getElementById("id_brightness");
  var brightnessVal = document.getElementById("brightness-val");

  function syncRange(range, hidden, label) {
    if (!range || !hidden) return;
    var v = range.value;
    hidden.value = v;
    if (label) label.textContent = v;
  }

  if (thresholdRange) {
    thresholdRange.addEventListener("input", function () {
      syncRange(thresholdRange, thresholdHidden, thresholdVal);
    });
    syncRange(thresholdRange, thresholdHidden, thresholdVal);
  }
  if (brightnessRange) {
    brightnessRange.addEventListener("input", function () {
      syncRange(brightnessRange, brightnessHidden, brightnessVal);
    });
    syncRange(brightnessRange, brightnessHidden, brightnessVal);
  }

  /* --- Tabs --- */
  var tabs = document.querySelectorAll(".tab[data-tab]");
  var panelUpload = document.getElementById("panel-upload");
  var panelCamera = document.getElementById("panel-camera");
  var btnSubmit = document.getElementById("btn-submit-upload");

  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      tabs.forEach(function (t) { t.classList.remove("tab--active"); });
      tab.classList.add("tab--active");
      var name = tab.dataset.tab;
      if (name === "camera") {
        if (panelUpload) { panelUpload.classList.remove("tab-panel--active"); }
        if (panelCamera) { panelCamera.classList.add("tab-panel--active"); }
        if (btnSubmit) btnSubmit.style.display = "none";
        startCamera();
      } else {
        if (panelUpload) { panelUpload.classList.add("tab-panel--active"); }
        if (panelCamera) { panelCamera.classList.remove("tab-panel--active"); }
        if (btnSubmit) btnSubmit.style.display = "";
        stopCamera();
      }
    });
  });

  /* --- Camera --- */
  var stream = null;
  var video = document.getElementById("camera-video");
  var canvas = document.getElementById("camera-canvas");
  var captureBtn = document.getElementById("camera-capture");

  function startCamera() {
    if (!video || stream) return;
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: "user", width: { ideal: 640 } }, audio: false })
      .then(function (s) {
        stream = s;
        video.srcObject = stream;
      })
      .catch(function () {
        if (window.bfrToast) bfrToast("تعذّر الوصول إلى الكاميرا.", "error");
      });
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach(function (t) { t.stop(); });
      stream = null;
    }
    if (video) video.srcObject = null;
  }

  if (captureBtn) {
    captureBtn.addEventListener("click", function () {
      if (!video || !canvas) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      var ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0);
      canvas.toBlob(function (blob) {
        if (!blob) return;
        var form = document.getElementById("verify-form");
        var fd = new FormData(form);
        fd.set("image", blob, "capture.jpg");
        var csrf = form.querySelector("[name=csrfmiddlewaretoken]").value;
        var apiUrl = form.dataset.apiUrl || "/api/verify/";

        if (window.bfrToast) bfrToast("جارٍ التحقق...", "info");

        fetch(apiUrl, {
          method: "POST",
          headers: { "X-CSRFToken": csrf },
          body: fd,
        })
          .then(function (res) { return res.json(); })
          .then(function (data) {
            var box = document.getElementById("camera-result");
            if (!box) {
              if (data.ok) location.reload();
              return;
            }
            box.style.display = "block";
            if (data.ok) {
              var statusClass = data.accepted ? "verify-result--ok" : "verify-result--fail";
              box.innerHTML =
                '<div class="verify-result ' + statusClass + '" style="margin-bottom:var(--sp-4)">' +
                '<p class="verify-result__decision">' + data.decision_ar + "</p>" +
                "</div>" +
                '<div class="metrics-grid">' +
                '<div class="metric"><div class="metric__label">الهوية</div><div class="metric__value">' + data.identity + "</div></div>" +
                '<div class="metric"><div class="metric__label">الثقة</div><div class="metric__value">' + data.confidence_pct + "%</div></div>" +
                '<div class="metric"><div class="metric__label">الزمن</div><div class="metric__value">' + data.latency_ms + " ms</div></div>" +
                "</div>" +
                (data.preview_b64
                  ? '<img src="data:image/jpeg;base64,' + data.preview_b64 + '" alt="preview" class="preview-figure__img mt-4">'
                  : "");
              if (window.bfrToast) bfrToast(data.accepted ? "تم قبول الهوية" : "تم رفض الهوية", data.accepted ? "success" : "error");
            } else {
              box.innerHTML = '<div class="alert alert--error">' + (data.error || "فشل التحقق") + "</div>";
              if (window.bfrToast) bfrToast(data.error || "فشل التحقق", "error");
            }
          })
          .catch(function () {
            if (window.bfrToast) bfrToast("خطأ في الاتصال بالخادم", "error");
          });
      }, "image/jpeg", 0.92);
    });
  }

  /* --- Form submit loading state --- */
  var verifyForm = document.getElementById("verify-form");
  if (verifyForm) {
    verifyForm.addEventListener("submit", function () {
      var btn = document.getElementById("btn-submit-upload");
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner" style="width:18px;height:18px;display:inline-block;vertical-align:middle;margin-left:8px"></span> جارٍ التحقق...';
      }
    });
  }

  window.addEventListener("beforeunload", stopCamera);
})();
