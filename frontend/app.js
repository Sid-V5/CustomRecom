/**
 * Mini-X Campus Social Recommender Web Client (app.js)
 * Clean, High-Craft Design System implementation.
 * Zero blue or yellow in background. Pure neutral light & dark modes.
 */

const API_BASE = "/api";

// Global Application State
const appState = {
  currentUserId: 0,
  users: [],
  activeTimelineTab: "for_you", // "for_you" | "following" | "history"
  activeNavTab: "feed",
  defenseActive: false,
  sliders: {
    mmr_lambda: 0.70,
    w_like: 0.60,
    w_reply: 1.20,
    w_skip: 0.40,
    lambda_age: 0.02,
  },
  activeFeed: [],
  likedPosts: new Set(),
  bookmarkedPosts: new Set(),
  activeTagFilter: null,
  activeSearchQuery: null,
  attackStatus: null,
  targetPostId: 42,
  debounceTimer: null,
};

// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initApp();
  initKeyboardShortcuts();
});

function initTheme() {
  const savedTheme = localStorage.getItem("mini_x_theme") || "dark";
  setTheme(savedTheme);
}

function toggleTheme() {
  const isDark = document.documentElement.classList.contains("dark");
  setTheme(isDark ? "light" : "dark");
}

function setTheme(theme) {
  const html = document.documentElement;
  const icon = document.getElementById("theme-icon");
  const label = document.getElementById("theme-label");

  if (theme === "dark") {
    html.classList.add("dark");
    if (icon) icon.textContent = "🌙";
    if (label) label.textContent = "Dark Mode";
  } else {
    html.classList.remove("dark");
    if (icon) icon.textContent = "☀️";
    if (label) label.textContent = "Light Mode";
  }
  localStorage.setItem("mini_x_theme", theme);
}

async function initApp() {
  try {
    // 1. Fetch user list
    const res = await fetch(`${API_BASE}/users`);
    const data = await res.json();
    appState.users = data.users || [];

    // Populate user dropdown in profile modal
    populateUserDropdown();

    // Set default user (prefer Alice CS_Undergrad or user 0)
    const defaultUser = appState.users.find(u => !u.is_cold_start && u.persona === "CS_Undergrad") || appState.users[0];
    if (defaultUser) {
      appState.currentUserId = defaultUser.user_id;
    }

    updateUserProfileUI();
    renderWhoToFollow();
    loadFeed();
    checkAttackStatus();
  } catch (err) {
    console.error("Initialization error:", err);
    showToast("Failed to connect to recommendation server.");
  }
}

function initKeyboardShortcuts() {
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeSettingsDrawer();
      closeProfileModal();
      closeBenchmarkModal();
    }
  });
}

function populateUserDropdown() {
  const select = document.getElementById("user-dropdown");
  if (!select) return;

  select.innerHTML = "";
  appState.users.forEach(u => {
    const opt = document.createElement("option");
    opt.value = u.user_id;
    opt.textContent = `#${u.user_id} - ${u.username} (${u.persona}${u.is_cold_start ? ' • Cold' : ''})`;
    select.appendChild(opt);
  });

  select.value = appState.currentUserId;
}

// ============================================================
// NAVIGATION & TABS
// ============================================================

function switchNavTab(tab) {
  appState.activeNavTab = tab;
  
  // Update left nav link highlights
  document.querySelectorAll(".nav-item").forEach(el => el.classList.remove("active"));
  const activeEl = document.getElementById(`nav-${tab}`);
  if (activeEl) activeEl.classList.add("active");

  if (tab === "feed") {
    switchTimelineTab("for_you");
  } else if (tab === "explore") {
    const input = document.getElementById("search-input");
    if (input) {
      input.focus();
      input.scrollIntoView({ behavior: 'smooth' });
    }
  } else if (tab === "history") {
    switchTimelineTab("history");
  } else if (tab === "profile") {
    openProfileModal();
  } else if (tab === "settings") {
    openSettingsDrawer();
  } else if (tab === "benchmarks") {
    openBenchmarkModal();
  }
}

function switchTimelineTab(tab) {
  appState.activeTimelineTab = tab;

  // Update sticky tab indicator
  document.querySelectorAll(".top-tab").forEach(el => el.classList.remove("active"));
  const tabEl = document.getElementById(`tab-${tab === "for_you" ? "for-you" : tab}`);
  if (tabEl) tabEl.classList.add("active");

  // Clear filters when switching top tabs
  appState.activeTagFilter = null;
  appState.activeSearchQuery = null;
  hideFilterBanner();

  if (tab === "history") {
    renderHistoryView();
  } else {
    loadFeed();
  }
}

// ============================================================
// FEED FETCHING & RENDERING
// ============================================================

async function loadFeed() {
  const feedList = document.getElementById("feed-list");
  if (!feedList) return;

  feedList.innerHTML = `
    <div class="text-center py-20 text-textMuted text-xs">
      <div class="inline-block animate-spin rounded-full h-6 w-6 border-2 border-textMuted border-t-transparent mb-3"></div>
      <p class="font-medium">Hydrating and ranking candidates...</p>
    </div>
  `;

  try {
    const s = appState.sliders;
    const mode = appState.activeTimelineTab === "following" ? "in_network" : "all";
    
    let url = `${API_BASE}/feed?user_id=${appState.currentUserId}&retrieval_mode=${mode}&mmr_lambda=${s.mmr_lambda}&w_like=${s.w_like}&w_reply=${s.w_reply}&w_skip=${s.w_skip}&lambda_age=${s.lambda_age}&defense_active=${appState.defenseActive}&top_k=15`;
    
    if (appState.activeTagFilter) {
      // If tag filter is active
      url = `${API_BASE}/search?q=${encodeURIComponent(appState.activeTagFilter)}&user_id=${appState.currentUserId}&top_k=15`;
    }

    const res = await fetch(url);
    const data = await res.json();

    appState.activeFeed = data.feed || [];
    renderFeed(appState.activeFeed);
    
    if (data.funnel_stats) {
      updateFunnelInspector(data.funnel_stats);
      
      const latEl = document.getElementById("header-latency");
      const candEl = document.getElementById("header-retrieved");
      if (latEl) latEl.textContent = `${data.funnel_stats.latency_ms} ms`;
      if (candEl) candEl.textContent = `${data.funnel_stats.retrieved_count}`;

      // Mode label
      const modeLabel = document.getElementById("header-pipeline-mode");
      if (modeLabel) {
        modeLabel.textContent = data.funnel_stats.is_cold_start 
          ? "Cold-Start Switch" 
          : (mode === "in_network" ? "In-Network Follows" : "Hybrid Heavy Ranker");
      }

      // Cold start banner
      const coldBanner = document.getElementById("cold-start-banner");
      if (coldBanner) {
        if (data.funnel_stats.is_cold_start) {
          coldBanner.classList.remove("hidden");
        } else {
          coldBanner.classList.add("hidden");
        }
      }
    }
  } catch (err) {
    console.error("Feed load error:", err);
    feedList.innerHTML = `
      <div class="p-8 text-center text-xs text-textMuted">
        <p class="font-semibold text-textPrimary mb-1">Failed to load feed recommendations.</p>
        <p>Ensure the FastAPI server is running on http://localhost:8000.</p>
      </div>
    `;
  }
}

function renderFeed(posts) {
  const feedList = document.getElementById("feed-list");
  if (!feedList) return;

  if (posts.length === 0) {
    feedList.innerHTML = `
      <div class="p-12 text-center text-textMuted text-xs">
        <p class="text-sm font-semibold text-textPrimary mb-1">No candidate posts to display.</p>
        <p>Try resetting filters or adjusting MMR diversity sliders.</p>
      </div>
    `;
    return;
  }

  feedList.innerHTML = "";
  posts.forEach((item, index) => {
    const card = createPostCardElement(item, index);
    feedList.appendChild(card);
  });
}

function createPostCardElement(item, index) {
  const card = document.createElement("article");
  card.id = `post-card-${item.post_id}`;
  card.className = "post-card space-y-2.5 relative";

  const isLiked = appState.likedPosts.has(item.post_id);
  const isBookmarked = appState.bookmarkedPosts.has(item.post_id);
  const expl = item.explanation || {};
  const probs = expl.probabilities || { 
    p_like: (item.p_like * 100).toFixed(1), 
    p_reply: (item.p_reply * 100).toFixed(1), 
    p_skip: (item.p_skip * 100).toFixed(1) 
  };
  const fw = expl.feature_weights || { network_affinity: 25, topic_alignment: 25, peer_collaborative: 25, discussion_velocity: 25 };

  // Format Hashtags
  const formattedContent = item.content.replace(/(#\w+)/g, '<span class="text-textPrimary hover:underline cursor-pointer font-semibold" onclick="filterFeedByTag(\'$1\')">$1</span>');
  const avatarInitials = (item.author_username || "U").substring(0, 2).toUpperCase();

  card.innerHTML = `
    <!-- Top Metadata Header -->
    <div class="flex items-start justify-between gap-2">
      <div class="flex items-center gap-2.5 min-w-0">
        <div class="w-9 h-9 rounded-full bg-surface border border-borderMedium flex items-center justify-center font-bold text-xs text-textPrimary shrink-0">
          ${avatarInitials}
        </div>
        <div class="min-w-0">
          <div class="flex items-center gap-1.5 flex-wrap">
            <span class="font-bold text-xs text-textPrimary truncate">${item.author_username}</span>
            <span class="text-[11px] text-textMuted font-mono">@${item.author_username.toLowerCase()}</span>
            <span class="text-textMuted text-[10px]">•</span>
            <span class="badge-clean text-[9px] py-0 px-1.5 font-normal">${item.department}</span>
          </div>
        </div>
      </div>

      <div class="flex items-center gap-1.5 shrink-0">
        <span class="badge-clean">
          ${formatSource(item.source)}
        </span>
        <span class="text-[10px] text-textMuted font-mono">#${index + 1}</span>
      </div>
    </div>

    <!-- Post Body Text -->
    <div class="text-[13.5px] leading-relaxed text-textPrimary font-sans pl-1">
      ${formattedContent}
    </div>

    <!-- Tag Pills -->
    <div class="flex flex-wrap gap-1 pl-1">
      ${(item.tags || []).map(t => `<span onclick="filterFeedByTag('${t}')" class="text-[10px] font-mono px-2 py-0.5 rounded-full bg-surface hover:bg-surfaceHover border border-borderSubtle text-textSecondary cursor-pointer transition">${t}</span>`).join('')}
    </div>

    <!-- Action Bar & Explainability Toggle -->
    <div class="flex items-center justify-between pt-1 text-xs text-textMuted">
      <div class="flex items-center gap-4">
        <!-- Reply -->
        <button onclick="handleInteraction(${item.post_id}, 'reply')" class="action-btn" title="Reply">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/></svg>
          <span id="reply-count-${item.post_id}" class="font-mono text-[11px]">${item.replies_count}</span>
        </button>

        <!-- Like -->
        <button onclick="handleInteraction(${item.post_id}, 'like')" class="action-btn ${isLiked ? 'liked' : ''}" title="Like">
          <svg class="w-4 h-4 ${isLiked ? 'liked-anim fill-rose-500 stroke-rose-500' : ''}" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/></svg>
          <span id="like-count-${item.post_id}" class="font-mono text-[11px]">${item.likes_count}</span>
        </button>

        <!-- Bookmark -->
        <button onclick="handleBookmark(${item.post_id})" class="action-btn ${isBookmarked ? 'bookmarked' : ''}" title="Bookmark">
          <svg class="w-4 h-4 ${isBookmarked ? 'fill-current' : ''}" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"/></svg>
        </button>

        <!-- Skip -->
        <button onclick="handleInteraction(${item.post_id}, 'skip')" class="action-btn" title="Skip / Dismiss">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
        </button>
      </div>

      <!-- Explainability Pill -->
      <button onclick="toggleExplainer(${item.post_id})" class="text-[11px] font-mono text-textSecondary hover:text-textPrimary flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-borderSubtle bg-surface hover:bg-surfaceHover transition">
        <span>Why this post?</span>
        <span id="expl-arrow-${item.post_id}" class="text-[9px] transition-transform">▼</span>
      </button>
    </div>

    <!-- Collapsible Explainability Drawer -->
    <div id="expl-drawer-${item.post_id}" class="hidden pt-3 border-t border-borderSubtle bg-surface rounded-xl p-3 text-xs space-y-2.5 animate-fadeIn">
      <div class="flex items-center justify-between">
        <span class="font-bold text-textPrimary">${expl.primary_reason || "Ranked candidate"}</span>
        <span class="font-mono text-[11px] text-textMuted">Score: <strong>${item.final_score.toFixed(4)}</strong></span>
      </div>

      <!-- Calibrated Probabilities -->
      <div class="grid grid-cols-3 gap-2 font-mono text-[11px] text-center">
        <div class="p-1.5 rounded-lg bg-surfaceElevated border border-borderSubtle">
          <div class="text-[10px] text-textMuted">P(Like)</div>
          <div class="font-bold text-textPrimary">${probs.p_like}%</div>
        </div>
        <div class="p-1.5 rounded-lg bg-surfaceElevated border border-borderSubtle">
          <div class="text-[10px] text-textMuted">P(Reply)</div>
          <div class="font-bold text-textPrimary">${probs.p_reply}%</div>
        </div>
        <div class="p-1.5 rounded-lg bg-surfaceElevated border border-borderSubtle">
          <div class="text-[10px] text-textMuted">P(Skip)</div>
          <div class="font-bold text-textPrimary">${probs.p_skip}%</div>
        </div>
      </div>

      <!-- Signal Attribution Breakdown -->
      <div class="space-y-1.5 text-[10px] font-mono pt-1">
        <div class="flex justify-between text-textMuted">
          <span>Social Network Affinity:</span>
          <span>${fw.network_affinity}%</span>
        </div>
        <div class="attr-bar-bg"><div class="attr-bar-fill" style="width: ${fw.network_affinity}%;"></div></div>

        <div class="flex justify-between text-textMuted">
          <span>Topic Alignment (TF-IDF):</span>
          <span>${fw.topic_alignment}%</span>
        </div>
        <div class="attr-bar-bg"><div class="attr-bar-fill" style="width: ${fw.topic_alignment}%;"></div></div>

        <div class="flex justify-between text-textMuted">
          <span>Collaborative Latent Match (iALS):</span>
          <span>${fw.peer_collaborative}%</span>
        </div>
        <div class="attr-bar-bg"><div class="attr-bar-fill" style="width: ${fw.peer_collaborative}%;"></div></div>
      </div>
    </div>
  `;

  return card;
}

function formatSource(source) {
  const map = {
    in_network: "Follows",
    als_latent: "iALS",
    content_tfidf: "Content",
    item_knn: "Item-kNN",
    trending_velocity: "Trending",
    case_based_search: "Search",
    cold_tag: "Knowledge",
  };
  return map[source] || source;
}

// ============================================================
// HISTORY VIEW (LIKED & BOOKMARKS - Image 1 Annotation)
// ============================================================

function renderHistoryView() {
  const feedList = document.getElementById("feed-list");
  if (!feedList) return;

  const historyIds = Array.from(new Set([...appState.likedPosts, ...appState.bookmarkedPosts]));

  if (historyIds.length === 0) {
    feedList.innerHTML = `
      <div class="p-16 text-center text-textMuted text-xs space-y-2">
        <p class="text-sm font-bold text-textPrimary">No saved posts yet.</p>
        <p>Like (❤️) or Bookmark (🔖) posts in your "For You" timeline to save them here.</p>
      </div>
    `;
    return;
  }

  // Filter from active feed or show placeholders
  const historyPosts = appState.activeFeed.filter(p => historyIds.includes(p.post_id));
  
  if (historyPosts.length === 0) {
    feedList.innerHTML = `
      <div class="p-16 text-center text-textMuted text-xs space-y-2">
        <p class="text-sm font-bold text-textPrimary">${historyIds.length} posts saved in session.</p>
        <p>Switch back to "For You" to browse more campus updates.</p>
      </div>
    `;
    return;
  }

  renderFeed(historyPosts);
}

// ============================================================
// INTERACTIONS & FEEDBACK
// ============================================================

async function handleInteraction(postId, action) {
  try {
    if (action === "like") {
      const isLiked = appState.likedPosts.has(postId);
      const countEl = document.getElementById(`like-count-${postId}`);
      if (isLiked) {
        appState.likedPosts.delete(postId);
        if (countEl) countEl.textContent = Math.max(0, parseInt(countEl.textContent) - 1);
      } else {
        appState.likedPosts.add(postId);
        if (countEl) countEl.textContent = parseInt(countEl.textContent) + 1;
      }
      updateLikedCountUI();
      renderFeed(appState.activeFeed);
    }

    if (action === "skip") {
      // Fade out card
      const card = document.getElementById(`post-card-${postId}`);
      if (card) {
        card.style.opacity = "0.3";
        setTimeout(() => card.remove(), 250);
      }
    }

    // Call REST endpoint
    await fetch(`${API_BASE}/interact`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: appState.currentUserId,
        post_id: postId,
        action: action,
        dwell_seconds: action === "skip" ? 2.0 : 25.0,
      }),
    });

    if (action === "reply") {
      const replyEl = document.getElementById(`reply-count-${postId}`);
      if (replyEl) replyEl.textContent = parseInt(replyEl.textContent) + 1;
      showToast("Reply logged to interaction matrix.");
    }
  } catch (err) {
    console.error("Interaction error:", err);
  }
}

function handleBookmark(postId) {
  if (appState.bookmarkedPosts.has(postId)) {
    appState.bookmarkedPosts.delete(postId);
    showToast("Removed from bookmarks");
  } else {
    appState.bookmarkedPosts.add(postId);
    showToast("Post bookmarked!");
  }
  updateLikedCountUI();
  renderFeed(appState.activeFeed);
}

function updateLikedCountUI() {
  const count = appState.likedPosts.size + appState.bookmarkedPosts.size;
  const el = document.getElementById("nav-liked-count");
  if (el) el.textContent = count;
}

function toggleExplainer(postId) {
  const drawer = document.getElementById(`expl-drawer-${postId}`);
  const arrow = document.getElementById(`expl-arrow-${postId}`);
  if (!drawer) return;

  const isHidden = drawer.classList.contains("hidden");
  if (isHidden) {
    drawer.classList.remove("hidden");
    if (arrow) arrow.style.transform = "rotate(180deg)";
  } else {
    drawer.classList.add("hidden");
    if (arrow) arrow.style.transform = "rotate(0deg)";
  }
}

// ============================================================
// CASE-BASED SEARCH & TAG FILTERS
// ============================================================

async function handleCaseSearch() {
  const input = document.getElementById("search-input");
  if (!input || !input.value.trim()) return;

  const query = input.value.trim();
  appState.activeSearchQuery = query;

  showFilterBanner(`Search: "${query}"`);
  const clearBtn = document.getElementById("search-clear-btn");
  if (clearBtn) clearBtn.classList.remove("hidden");

  const feedList = document.getElementById("feed-list");
  feedList.innerHTML = `
    <div class="text-center py-20 text-textMuted text-xs">
      <div class="inline-block animate-spin rounded-full h-6 w-6 border-2 border-textMuted border-t-transparent mb-2"></div>
      <p>Running Case-Based Vector Search (TF-IDF)...</p>
    </div>
  `;

  try {
    const res = await fetch(`${API_BASE}/search?q=${encodeURIComponent(query)}&user_id=${appState.currentUserId}&top_k=15`);
    const data = await res.json();
    renderFeed(data.feed || []);
  } catch (err) {
    console.error("Search error:", err);
  }
}

function clearSearch() {
  const input = document.getElementById("search-input");
  if (input) input.value = "";
  const clearBtn = document.getElementById("search-clear-btn");
  if (clearBtn) clearBtn.classList.add("hidden");

  appState.activeSearchQuery = null;
  clearActiveFilter();
}

function filterFeedByTag(tag) {
  appState.activeTagFilter = tag;
  showFilterBanner(`Tag: ${tag}`);
  loadFeed();
}

function showFilterBanner(label) {
  const banner = document.getElementById("active-filter-banner");
  const labelEl = document.getElementById("filter-label");
  if (banner && labelEl) {
    labelEl.textContent = label;
    banner.classList.remove("hidden");
  }
}

function hideFilterBanner() {
  const banner = document.getElementById("active-filter-banner");
  if (banner) banner.classList.add("hidden");
}

function clearActiveFilter() {
  appState.activeTagFilter = null;
  appState.activeSearchQuery = null;
  hideFilterBanner();
  loadFeed();
}

// ============================================================
// PROFILE & PERSONA IMPERSONATION (Image 1 Annotation)
// ============================================================

function openProfileModal() {
  const modal = document.getElementById("profile-modal");
  if (modal) modal.classList.remove("hidden");
  updateModalUserStats();
}

function closeProfileModal() {
  const modal = document.getElementById("profile-modal");
  if (modal) modal.classList.add("hidden");
}

function impersonatePreset(name) {
  let target = null;
  if (name === "Alice") {
    target = appState.users.find(u => !u.is_cold_start && u.persona === "CS_Undergrad");
  } else if (name === "Dr. Vance") {
    target = appState.users.find(u => !u.is_cold_start && u.persona === "Bio_Researcher");
  } else if (name === "Charlie") {
    target = appState.users.find(u => u.is_cold_start);
  } else if (name === "Robotics Club") {
    target = appState.users.find(u => u.persona === "Campus_Club");
  }

  if (target) {
    impersonateUser(target.user_id);
    closeProfileModal();
  }
}

function impersonateUser(userId) {
  appState.currentUserId = parseInt(userId);
  
  // Update select value
  const select = document.getElementById("user-dropdown");
  if (select) select.value = appState.currentUserId;

  updateUserProfileUI();
  updateModalUserStats();
  loadFeed();
  checkAttackStatus();
  showToast(`Switched active user to #${appState.currentUserId}`);
}

function updateUserProfileUI() {
  const user = appState.users.find(u => u.user_id === appState.currentUserId);
  if (!user) return;

  const nameEl = document.getElementById("user-display-name");
  const handleEl = document.getElementById("user-handle");
  const avatarEl = document.getElementById("user-avatar");
  const badgeEl = document.getElementById("cold-start-badge");

  if (nameEl) nameEl.textContent = user.username;
  if (handleEl) handleEl.textContent = `@${user.username.toLowerCase()}`;
  if (avatarEl) avatarEl.textContent = user.username.substring(0, 2).toUpperCase();

  if (badgeEl) {
    if (user.is_cold_start) {
      badgeEl.classList.remove("hidden");
    } else {
      badgeEl.classList.add("hidden");
    }
  }
}

async function updateModalUserStats() {
  try {
    const res = await fetch(`${API_BASE}/user/${appState.currentUserId}`);
    const data = await res.json();

    const deptEl = document.getElementById("modal-user-dept");
    const countEl = document.getElementById("modal-user-interactions");
    const tagsEl = document.getElementById("modal-user-tags");

    if (deptEl) deptEl.textContent = data.department || "--";
    if (countEl) countEl.textContent = `${data.recent_interactions ? data.recent_interactions.length : 0} interactions recorded`;
    if (tagsEl) tagsEl.textContent = (data.preferred_tags || []).join(", ") || "General";
  } catch (err) {
    console.error("Profile stats error:", err);
  }
}

function renderWhoToFollow() {
  const container = document.getElementById("who-to-follow-list");
  if (!container) return;

  const suggestions = [
    { name: "Robotics Club", handle: "@robotics_club", dept: "Clubs", initials: "RC" },
    { name: "AI Research Lab", handle: "@ai_lab", dept: "Computer Science", initials: "AI" },
    { name: "Dr. Vance", handle: "@vance_bio", dept: "Biology", initials: "DV" },
  ];

  container.innerHTML = suggestions.map(s => `
    <div class="flex items-center justify-between text-xs">
      <div class="flex items-center gap-2">
        <div class="w-8 h-8 rounded-full bg-surfaceElevated border border-borderMedium flex items-center justify-center font-bold text-[10px] text-textPrimary">${s.initials}</div>
        <div>
          <p class="font-bold text-textPrimary">${s.name}</p>
          <p class="text-[10px] text-textMuted font-mono">${s.handle}</p>
        </div>
      </div>
      <button onclick="toggleFollow(this)" class="bg-accentContrast text-[var(--accent-contrast-text)] text-[11px] font-semibold px-3 py-1 rounded-full hover:opacity-90 transition">Follow</button>
    </div>
  `).join('');
}

function toggleFollow(btn) {
  if (btn.textContent === "Follow") {
    btn.textContent = "Following";
    btn.classList.remove("bg-accentContrast", "text-[var(--accent-contrast-text)]");
    btn.classList.add("bg-surfaceHover", "text-textPrimary", "border", "border-borderSubtle");
    showToast("Added to your In-Network follow graph!");
  } else {
    btn.textContent = "Follow";
    btn.classList.add("bg-accentContrast", "text-[var(--accent-contrast-text)]");
    btn.classList.remove("bg-surfaceHover", "text-textPrimary", "border", "border-borderSubtle");
    showToast("Unfollowed.");
  }
  loadFeed();
}

// ============================================================
// SETTINGS DRAWER (SLIDERS, SANDBOX, FUNNEL - Image 2 Red Circle)
// ============================================================

function openSettingsDrawer(initialTab = "sliders") {
  const backdrop = document.getElementById("settings-drawer-backdrop");
  const panel = document.getElementById("settings-drawer-panel");

  if (backdrop && panel) {
    backdrop.classList.remove("opacity-0", "pointer-events-none");
    panel.classList.remove("translate-x-full");
    switchDrawerTab(initialTab);
  }
}

function closeSettingsDrawer() {
  const backdrop = document.getElementById("settings-drawer-backdrop");
  const panel = document.getElementById("settings-drawer-panel");

  if (backdrop && panel) {
    backdrop.classList.add("opacity-0", "pointer-events-none");
    panel.classList.add("translate-x-full");
  }
}

function switchDrawerTab(tab) {
  // Update tabs
  ["sliders", "sandbox", "funnel"].forEach(t => {
    const btn = document.getElementById(`drawer-tab-${t}`);
    const sec = document.getElementById(`drawer-section-${t}`);
    if (btn) {
      if (t === tab) {
        btn.classList.add("border-accentContrast", "text-textPrimary", "font-bold");
        btn.classList.remove("border-transparent", "text-textMuted");
      } else {
        btn.classList.remove("border-accentContrast", "text-textPrimary", "font-bold");
        btn.classList.add("border-transparent", "text-textMuted");
      }
    }
    if (sec) {
      if (t === tab) sec.classList.remove("hidden");
      else sec.classList.add("hidden");
    }
  });
}

function handleSliderChange() {
  const mmr = parseFloat(document.getElementById("slider-mmr").value);
  const like = parseFloat(document.getElementById("slider-like").value);
  const reply = parseFloat(document.getElementById("slider-reply").value);
  const skip = parseFloat(document.getElementById("slider-skip").value);
  const age = parseFloat(document.getElementById("slider-age").value);

  appState.sliders.mmr_lambda = mmr;
  appState.sliders.w_like = like;
  appState.sliders.w_reply = reply;
  appState.sliders.w_skip = skip;
  appState.sliders.lambda_age = age;

  document.getElementById("slider-val-mmr").textContent = mmr.toFixed(2);
  document.getElementById("slider-val-like").textContent = like.toFixed(2);
  document.getElementById("slider-val-reply").textContent = reply.toFixed(2);
  document.getElementById("slider-val-skip").textContent = skip.toFixed(2);
  document.getElementById("slider-val-age").textContent = age.toFixed(3);

  clearTimeout(appState.debounceTimer);
  appState.debounceTimer = setTimeout(() => {
    loadFeed();
  }, 200);
}

function resetSliders() {
  appState.sliders = {
    mmr_lambda: 0.70,
    w_like: 0.60,
    w_reply: 1.20,
    w_skip: 0.40,
    lambda_age: 0.02,
  };

  document.getElementById("slider-mmr").value = 0.70;
  document.getElementById("slider-like").value = 0.60;
  document.getElementById("slider-reply").value = 1.20;
  document.getElementById("slider-skip").value = 0.40;
  document.getElementById("slider-age").value = 0.02;

  document.getElementById("slider-val-mmr").textContent = "0.70";
  document.getElementById("slider-val-like").textContent = "0.60";
  document.getElementById("slider-val-reply").textContent = "1.20";
  document.getElementById("slider-val-skip").textContent = "0.40";
  document.getElementById("slider-val-age").textContent = "0.020";

  loadFeed();
  showToast("Sliders reset to baseline defaults.");
}

function updateFunnelInspector(stats) {
  if (!stats) return;

  const corpusEl = document.getElementById("funnel-corpus");
  const srcEl = document.getElementById("funnel-sourced");
  const hydEl = document.getElementById("funnel-hydrated");
  const filtEl = document.getElementById("funnel-filtered");
  const feedEl = document.getElementById("funnel-feed");

  if (corpusEl) corpusEl.textContent = stats.corpus_count.toLocaleString();
  if (srcEl) srcEl.textContent = stats.retrieved_count;
  if (hydEl) hydEl.textContent = stats.hydrated_count;
  if (filtEl) filtEl.textContent = stats.filtered_count;
  if (feedEl) feedEl.textContent = stats.feed_count;

  // Update progress bars
  const total = stats.corpus_count || 3500;
  const barSourced = document.getElementById("funnel-bar-sourced");
  const barHydrated = document.getElementById("funnel-bar-hydrated");
  const barFiltered = document.getElementById("funnel-bar-filtered");

  if (barSourced) barSourced.style.width = `${Math.min(100, (stats.retrieved_count / total) * 100 * 5)}%`;
  if (barHydrated) barHydrated.style.width = `${Math.min(100, (stats.hydrated_count / total) * 100 * 5)}%`;
  if (barFiltered) barFiltered.style.width = `${Math.min(100, (stats.filtered_count / total) * 100 * 5)}%`;

  // Detailed filter breakdown
  const chips = document.getElementById("funnel-filter-chips");
  if (chips && stats.filter_breakdown) {
    const fb = stats.filter_breakdown;
    chips.innerHTML = `
      <div class="flex justify-between"><span>Dropped (Already Seen):</span><span>${fb.dropped_seen || 0}</span></div>
      <div class="flex justify-between"><span>Dropped (Author Spam >5):</span><span>${fb.dropped_author_spam || 0}</span></div>
      <div class="flex justify-between"><span>Dropped (Self / Blocked):</span><span>${(fb.dropped_self || 0) + (fb.dropped_blocked || 0)}</span></div>
      <div class="flex justify-between"><span>Dropped (Age Cutoff >7d):</span><span>${fb.dropped_age || 0}</span></div>
      <div class="flex justify-between"><span>Dropped (Shilling Botnet):</span><span>${fb.dropped_shilling || 0}</span></div>
    `;
  }
}

// ============================================================
// SECURITY SANDBOX (SHILLING ATTACK & DEFENSE)
// ============================================================

async function injectAttack(type) {
  try {
    const res = await fetch(`${API_BASE}/attack/inject`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_post_id: appState.targetPostId,
        attack_type: type,
      }),
    });
    const data = await res.json();
    showToast(`Injected ${type.toUpperCase()} like-bomb on Post #${appState.targetPostId}!`);
    checkAttackStatus();
    loadFeed();
  } catch (err) {
    console.error("Attack injection error:", err);
  }
}

async function resetAttack() {
  try {
    await fetch(`${API_BASE}/attack/reset`, { method: "POST" });
    showToast("Attack botnet cleared. Organic state restored.");
    checkAttackStatus();
    loadFeed();
  } catch (err) {
    console.error("Attack reset error:", err);
  }
}

async function checkAttackStatus() {
  try {
    const res = await fetch(`${API_BASE}/attack/status?user_id=${appState.currentUserId}&target_post_id=${appState.targetPostId}`);
    const data = await res.json();
    appState.attackStatus = data;

    const rankUnprot = document.getElementById("sandbox-attack-rank");
    const statusText = document.getElementById("sandbox-bot-status");
    const preview = document.getElementById("sandbox-target-preview");

    if (data.target_post_snippet && preview) {
      preview.textContent = `"${data.target_post_snippet}"`;
    }

    if (rankUnprot) {
      rankUnprot.textContent = data.target_post_rank_unprotected 
        ? `#${data.target_post_rank_unprotected}` 
        : "--";
    }

    if (statusText) {
      if (data.is_flagged_as_botnet) {
        statusText.innerHTML = `<span class="text-rose-500 font-bold">⚠️ BOTNET DETECTED</span> (Anomaly Score: ${(data.anomaly_score * 100).toFixed(0)}%)`;
      } else {
        statusText.textContent = "Status: Organic (No bots detected)";
      }
    }
  } catch (err) {
    console.error("Attack status check error:", err);
  }
}

function toggleDefenseFromDrawer(checked) {
  appState.defenseActive = checked;
  
  // Sync header / sidebar badges
  const badge = document.getElementById("sidebar-defense-badge");
  if (badge) {
    badge.textContent = checked ? "Enabled 🛡️" : "Disabled";
    badge.className = checked ? "font-mono text-emerald-500 font-bold" : "font-mono text-textMuted";
  }

  loadFeed();
  checkAttackStatus();
  showToast(checked ? "Shilling Defense Filter Active 🛡️" : "Defense Filter Disabled");
}

// ============================================================
// ACADEMIC BENCHMARKS & EVALUATION MODAL
// ============================================================

async function openBenchmarkModal() {
  const modal = document.getElementById("benchmark-modal");
  if (modal) modal.classList.remove("hidden");
  loadBenchmarks();
}

function closeBenchmarkModal() {
  const modal = document.getElementById("benchmark-modal");
  if (modal) modal.classList.add("hidden");
}

async function loadBenchmarks() {
  try {
    const res = await fetch(`${API_BASE}/benchmark/summary`);
    const data = await res.json();

    // 1. Primary baselines
    const mainBody = document.getElementById("benchmark-table-body");
    if (mainBody && data.main_benchmarks) {
      mainBody.innerHTML = data.main_benchmarks.map(m => `
        <tr class="hover:bg-surfaceHover">
          <td class="p-2.5 font-bold ${m.model.includes('Ours') ? 'text-textPrimary' : 'text-textSecondary'}">${m.model}</td>
          <td class="p-2.5">${(m.ndcg || m['ndcg@10'] || 0).toFixed(4)}</td>
          <td class="p-2.5">${(m.precision || m['p@10'] || 0).toFixed(4)}</td>
          <td class="p-2.5">${(m.recall || m['r@10'] || 0).toFixed(4)}</td>
          <td class="p-2.5">${(m.coverage || m.coverage_pct || 0).toFixed(1)}%</td>
          <td class="p-2.5">${(m.ild || m.ild_diversity || 0).toFixed(4)}</td>
          <td class="p-2.5">${m.rmse ? m.rmse.toFixed(4) : '--'}</td>
        </tr>
      `).join('');
    }

    // 2. Cold Start split
    const coldBody = document.getElementById("cold-benchmark-table-body");
    if (coldBody && data.cold_start_benchmarks) {
      coldBody.innerHTML = data.cold_start_benchmarks.map(m => `
        <tr class="hover:bg-surfaceHover">
          <td class="p-2.5 font-bold text-textPrimary">${m.model}</td>
          <td class="p-2.5">${(m.ndcg || m['ndcg@10'] || 0).toFixed(4)}</td>
          <td class="p-2.5">${(m.precision || m['p@10'] || 0).toFixed(4)}</td>
          <td class="p-2.5">${(m.recall || m['r@10'] || 0).toFixed(4)}</td>
          <td class="p-2.5">${(m.coverage || m.coverage_pct || 0).toFixed(1)}%</td>
          <td class="p-2.5">${(m.ild || m.ild_diversity || 0).toFixed(4)}</td>
        </tr>
      `).join('');
    }

    // 3. Ablations
    const ablBody = document.getElementById("ablation-table-body");
    if (ablBody && data.ablations) {
      ablBody.innerHTML = data.ablations.map(m => `
        <tr class="hover:bg-surfaceHover">
          <td class="p-2.5 font-bold text-textPrimary">${m.model}</td>
          <td class="p-2.5">${(m.ndcg || m['ndcg@10'] || 0).toFixed(4)}</td>
          <td class="p-2.5">${(m.precision || m['p@10'] || 0).toFixed(4)}</td>
          <td class="p-2.5">${(m.coverage || m.coverage_pct || 0).toFixed(1)}%</td>
          <td class="p-2.5">${(m.ild || m.ild_diversity || 0).toFixed(4)}</td>
        </tr>
      `).join('');
    }
  } catch (err) {
    console.error("Benchmarks load error:", err);
  }
}

// ============================================================
// TOAST NOTIFICATIONS
// ============================================================

function showToast(message) {
  const toast = document.getElementById("toast");
  if (!toast) return;

  toast.textContent = message;
  toast.classList.remove("translate-y-16", "opacity-0");
  toast.classList.add("translate-y-0", "opacity-100");

  setTimeout(() => {
    toast.classList.remove("translate-y-0", "opacity-100");
    toast.classList.add("translate-y-16", "opacity-0");
  }, 2400);
}
