/* Theme toggle + sidebar mobile + toast system */
(function () {
  /* --- Theme toggle --- */
  var toggle = document.getElementById("theme-toggle");
  var html = document.documentElement;

  function applyTheme(theme) {
    if (theme === "dark") {
      html.setAttribute("data-theme", "dark");
    } else {
      html.removeAttribute("data-theme");
    }
    updateIcons(theme);
    localStorage.setItem("bfr-theme", theme);
  }

  function currentTheme() {
    return html.getAttribute("data-theme") === "dark" ? "dark" : "light";
  }

  function updateIcons(theme) {
    var lightIcon = document.querySelector(".theme-icon--light");
    var darkIcon = document.querySelector(".theme-icon--dark");
    if (!lightIcon || !darkIcon) return;
    if (theme === "dark") {
      lightIcon.style.display = "none";
      darkIcon.style.display = "block";
    } else {
      lightIcon.style.display = "block";
      darkIcon.style.display = "none";
    }
  }

  if (toggle) {
    toggle.addEventListener("click", function () {
      applyTheme(currentTheme() === "dark" ? "light" : "dark");
    });
  }

  updateIcons(currentTheme());

  /* --- Mobile sidebar --- */
  var sidebar = document.getElementById("sidebar");
  var overlay = document.getElementById("sidebar-overlay");
  var menuBtn = document.getElementById("mobile-menu-btn");

  function openSidebar() {
    if (sidebar) sidebar.classList.add("sidebar--open");
    if (overlay) overlay.classList.add("sidebar-overlay--visible");
    document.body.style.overflow = "hidden";
  }

  function closeSidebar() {
    if (sidebar) sidebar.classList.remove("sidebar--open");
    if (overlay) overlay.classList.remove("sidebar-overlay--visible");
    document.body.style.overflow = "";
  }

  if (menuBtn) menuBtn.addEventListener("click", openSidebar);
  if (overlay) overlay.addEventListener("click", closeSidebar);

  /* --- Toast system --- */
  window.bfrToast = function (message, type) {
    type = type || "info";
    var container = document.getElementById("toast-container");
    if (!container) return;
    var toast = document.createElement("div");
    toast.className = "toast toast--" + type;
    var icon = "";
    if (type === "success") {
      icon = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>';
    } else if (type === "error") {
      icon = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>';
    } else {
      icon = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>';
    }
    toast.innerHTML = icon + "<span>" + message + "</span>";
    container.appendChild(toast);
    setTimeout(function () {
      toast.classList.add("toast--out");
      setTimeout(function () { toast.remove(); }, 300);
    }, 4000);
  };

  /* --- Table search --- */
  document.querySelectorAll("[data-table-search]").forEach(function (input) {
    var tableId = input.getAttribute("data-table-search");
    var table = document.getElementById(tableId);
    if (!table) return;
    input.addEventListener("input", function () {
      var q = input.value.toLowerCase();
      var rows = table.querySelectorAll("tbody tr");
      rows.forEach(function (row) {
        var text = row.textContent.toLowerCase();
        row.style.display = text.indexOf(q) > -1 ? "" : "none";
      });
    });
  });
})();
