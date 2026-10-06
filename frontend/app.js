/**
 * Mini-X Campus Social Recommender Web Client (app.js)
 * Manages dynamic feed generation, persona switching, pipeline telemetry,
 * real-time interaction feedback, attack sandbox, and offline benchmark tables.
 */

const API_BASE = "/api";

// Global Application State
const appState = {
  currentUserId: 0,
  users: [],
  activeTab: "feed",
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
  debounceTimer: null,
};

// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
  initApp();
});

async function initApp() {
  try {
    // 1. Fetch user list
    const res = await fetch(`${API_BASE}/users`);
    const data = await res.json();
    appState.users = data.users || [];

    // Populate user dropdown
    populateUserDropdown();

    // Setup defense toggle listeners
    const defHeader = document.getElementById("defense-toggle-header");
    if (defHeader) {
      defHeader.addEventListener("change", (e) => {
        appState.defenseActive = e.target.checked;
        loadFeed();
        checkAttackStatus();
      });
    }

    // Set default user (prefer warm user)
    const defaultUser = appState.users.find(u => !u.is_cold_start && u.persona === "CS_Undergrad") || appState.users[0];
    if (defaultUser) {
      appState.currentUserId = defaultUser.user_id;
    }

    updateUserProfileUI();
    loadFeed();
    checkAttackStatus();
  } catch (err) {
    console.error("Initialization error:", err);
    showToast("Failed to connect to recommendation server.");
  }
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
// FEED FETCHING & RENDERING
// ============================================================

async function loadFeed() {
  const feedList = document.getElementById("feed-list");
  if (!feedList) return;

  feedList.innerHTML = `
    <div class="text-center py-12 text-slate-500 text-sm">
      <div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-sky-500 mb-2"></div>
      <p>Hydrating and ranking candidates...</p>
    </div>
  `;

  try {
    const s = appState.sliders;
    const url = `${API_BASE}/feed?user_id=${appState.currentUserId}&mmr_lambda=${s.mmr_lambda}&w_like=${s.w_like}&w_reply=${s.w_reply}&w_skip=${s.w_skip}&lambda_age=${s.lambda_age}&defense_active=${appState.defenseActive}`;

    const res = await fetch(url);
    const data = await res.json();

    appState.activeFeed = data.feed || [];
    renderFeed(appState.activeFeed);
    updateFunnelInspector(data.funnel_stats);

    // Update header telemetry
    if (data.funnel_stats) {
      const latEl = document.getElementById("header-latency");
      const candEl = document.getElementById("header-retrieved");
      if (latEl) latEl.textContent = `${data.funnel_stats.latency_ms} ms`;
      if (candEl) candEl.textContent = `${data.funnel_stats.retrieved_count}`;

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
      <div class="bg-rose-500/10 border border-rose-500/30 rounded-xl p-4 text-xs text-rose-300 text-center">
        Error loading feed recommendations. Please check server logs.
      </div>
    `;
  }
}

function renderFeed(posts) {
  const feedList = document.getElementById("feed-list");
  if (!feedList) return;

  if (posts.length === 0) {
    feedList.innerHTML = `
      <div class="bg-[#121824] border border-[#243046] rounded-2xl p-8 text-center text-slate-400 text-xs">
        <p class="text-sm font-semibold text-slate-300 mb-1">No candidate posts survived filters.</p>
        <p>Try broadening interest tags or resetting sliders.</p>
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
  card.className = "post-card bg-[#121824] border border-[#243046] rounded-2xl p-4.5 space-y-3 transition relative";

  const isLiked = appState.likedPosts.has(item.post_id);
  const expl = item.explanation || {};
  const probs = expl.probabilities || { p_like: (item.p_like * 100).toFixed(1), p_reply: (item.p_reply * 100).toFixed(1), p_skip: (item.p_skip * 100).toFixed(1) };
  const fw = expl.feature_weights || { network_affinity: 25, topic_alignment: 25, peer_collaborative: 25, discussion_velocity: 25 };

  // Source Badge Color
  const sourceColors = {
    in_network: "bg-blue-500/20 text-blue-400 border-blue-500/30",
    als_latent: "bg-purple-500/20 text-purple-400 border-purple-500/30",
    content_tfidf: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
    item_knn: "bg-indigo-500/20 text-indigo-400 border-indigo-500/30",
    cold_tag: "bg-amber-500/20 text-amber-400 border-amber-500/30",
    trending_velocity: "bg-pink-500/20 text-pink-400 border-pink-500/30",
    case_based_search: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
  };
  const sourceBadgeClass = sourceColors[item.source] || "bg-slate-700 text-slate-300";

  // Hashtags formatted with clickable links
  const formattedContent = item.content.replace(/(#\w+)/g, '<span class="text-sky-400 hover:underline cursor-pointer font-medium" onclick="filterFeedByTag(\'$1\')">$1</span>');

  const avatarInitials = (item.author_username || "U").substring(0, 2).toUpperCase();

  card.innerHTML = `
    <!-- Top Header: Author + Source Badge + Rank -->
    <div class="flex items-start justify-between gap-2">
      <div class="flex items-center gap-2.5">
        <div class="w-10 h-10 rounded-full bg-gradient-to-tr from-slate-700 to-slate-800 border border-[#243046] flex items-center justify-center font-bold text-xs text-sky-400">
          ${avatarInitials}
        </div>
        <div>
          <div class="flex items-center gap-2">
            <span class="font-bold text-xs text-white">${item.author_username}</span>
            <span class="text-[11px] text-slate-400 font-mono">@${item.author_username.toLowerCase()}</span>
          </div>
          <span class="text-[10px] text-slate-400">${item.department}</span>
        </div>
      </div>

      <div class="flex items-center gap-1.5">
        <span class="text-[10px] font-mono px-2 py-0.5 rounded-full border ${sourceBadgeClass}">
          ${formatSource(item.source)}
        </span>
        <span class="text-[10px] text-slate-500 font-mono font-semibold">#${index + 1}</span>
      </div>
    </div>

    <!-- Post Body Content -->
    <div class="text-xs text-slate-200 leading-relaxed pl-1 pt-1">
      ${formattedContent}
    </div>

    <!-- Tags Row -->
    <div class="flex flex-wrap gap-1 pl-1">
      ${(item.tags || []).map(t => `<span onclick="filterFeedByTag('${t}')" class="text-[10px] bg-[#161f30] hover:bg-[#1c273c] text-slate-300 px-2 py-0.5 rounded-md cursor-pointer border border-[#243046] transition">${t}</span>`).join('')}
    </div>

    <!-- Engagement Buttons Row -->
    <div class="flex items-center justify-between pt-2 border-t border-[#243046] text-xs text-slate-400">
      <div class="flex items-center gap-6">
        <!-- Like -->
        <button onclick="handleInteraction(${item.post_id}, 'like')" class="flex items-center gap-1.5 hover:text-rose-400 transition group ${isLiked ? 'text-rose-500 font-bold' : ''}">
          <span class="${isLiked ? 'liked-anim' : ''}">${isLiked ? '❤️' : '🤍'}</span>
          <span id="like-count-${item.post_id}" class="font-mono text-[11px]">${item.likes_count}</span>
        </button>

        <!-- Reply -->
        <button onclick="handleInteraction(${item.post_id}, 'reply')" class="flex items-center gap-1.5 hover:text-sky-400 transition">
          <span>💬</span>
          <span class="font-mono text-[11px]">${item.replies_count}</span>
        </button>

        <!-- Skip / Dismiss -->
        <button onclick="handleInteraction(${item.post_id}, 'skip')" class="flex items-center gap-1 hover:text-amber-400 transition text-[11px]" title="Skip / Not interested">
          <span>✕</span>
          <span class="text-[10px]">Skip</span>
        </button>
      </div>

      <!-- Expandable "Why this post?" pill -->
      <button onclick="toggleExplainer(${item.post_id})" class="text-[11px] text-sky-400 hover:text-sky-300 font-medium flex items-center gap-1 bg-[#161f30] hover:bg-[#1c273c] border border-[#243046] px-2.5 py-1 rounded-lg transition">
        <span>💡 Why this post?</span>
        <span id="expl-arrow-${item.post_id}" class="transition-transform">▼</span>
      </button>
    </div>

    <!-- Expandable Explainability Drawer -->
    <div id="expl-drawer-${item.post_id}" class="hidden pt-3 mt-2 border-t border-[#243046] bg-[#0e1420] rounded-xl p-3 text-xs space-y-2.5 animate-fadeIn">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-1.5">
          <span class="text-sm">🎯</span>
          <span class="font-semibold text-sky-300">${expl.primary_reason || 'Personalized for your campus interests'}</span>
        </div>
        <span class="font-mono text-[11px] text-emerald-400 font-bold">Score: ${item.final_score.toFixed(4)}</span>
      </div>

      <!-- Predicted Multi-Action Probabilities -->
      <div class="grid grid-cols-3 gap-2 bg-[#161f30] p-2 rounded-lg text-center font-mono text-[11px]">
        <div>
          <div class="text-slate-400 text-[10px]">P(Like)</div>
          <div class="font-bold text-sky-400">${probs.p_like}%</div>
        </div>
        <div>
          <div class="text-slate-400 text-[10px]">P(Reply)</div>
          <div class="font-bold text-purple-400">${probs.p_reply}%</div>
        </div>
        <div>
          <div class="text-slate-400 text-[10px]">P(Skip)</div>
          <div class="font-bold text-rose-400">${probs.p_skip}%</div>
        </div>
      </div>

      <!-- Feature Attribution Breakdown Bars -->
      <div class="space-y-1.5 pt-1 text-[11px]">
        <div class="text-slate-400 text-[10px] font-semibold uppercase tracking-wider">Signal Attribution Decomposition:</div>
        <div>
          <div class="flex justify-between text-slate-300 text-[10px] mb-0.5">
            <span>In-Network Follow Affinity:</span>
            <span class="font-mono text-blue-400">${fw.network_affinity}%</span>
          </div>
          <div class="w-full bg-[#161f30] rounded-full h-1">
            <div class="bg-blue-500 h-1 rounded-full" style="width: ${fw.network_affinity}%"></div>
          </div>
        </div>

        <div>
          <div class="flex justify-between text-slate-300 text-[10px] mb-0.5">
            <span>Topic / Tag Cosine Alignment:</span>
            <span class="font-mono text-emerald-400">${fw.topic_alignment}%</span>
          </div>
          <div class="w-full bg-[#161f30] rounded-full h-1">
            <div class="bg-emerald-500 h-1 rounded-full" style="width: ${fw.topic_alignment}%"></div>
          </div>
        </div>

        <div>
          <div class="flex justify-between text-slate-300 text-[10px] mb-0.5">
            <span>Peer Collaborative Similarity (iALS / kNN):</span>
            <span class="font-mono text-purple-400">${fw.peer_collaborative}%</span>
          </div>
          <div class="w-full bg-[#161f30] rounded-full h-1">
            <div class="bg-purple-500 h-1 rounded-full" style="width: ${fw.peer_collaborative}%"></div>
          </div>
        </div>

        <div>
          <div class="flex justify-between text-slate-300 text-[10px] mb-0.5">
            <span>Social Discussion Velocity:</span>
            <span class="font-mono text-rose-400">${fw.discussion_velocity}%</span>
          </div>
          <div class="w-full bg-[#161f30] rounded-full h-1">
            <div class="bg-rose-500 h-1 rounded-full" style="width: ${fw.discussion_velocity}%"></div>
          </div>
        </div>
      </div>
    </div>
  `;

  return card;
}

function formatSource(src) {
  switch (src) {
    case "in_network": return "In-Network";
    case "als_latent": return "ALS Latent";
    case "content_tfidf": return "Content Match";
    case "item_knn": return "Item-kNN";
    case "cold_tag": return "Tag Match";
    case "trending_velocity": return "Trending / Velocity";
    case "case_based_search": return "Corpus Search";
    default: return src;
  }
}

function toggleExplainer(postId) {
  const drawer = document.getElementById(`expl-drawer-${postId}`);
  const arrow = document.getElementById(`expl-arrow-${postId}`);
  if (!drawer) return;

  if (drawer.classList.contains("hidden")) {
    drawer.classList.remove("hidden");
    if (arrow) arrow.style.transform = "rotate(180deg)";
  } else {
    drawer.classList.add("hidden");
    if (arrow) arrow.style.transform = "rotate(0deg)";
  }
}

// ============================================================
// INTERACTIONS & FEEDBACK LOGGING
// ============================================================

async function handleInteraction(postId, action) {
  try {
    const res = await fetch(`${API_BASE}/interact`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: appState.currentUserId,
        post_id: postId,
        action: action,
        dwell_seconds: action === "like" ? 30.0 : (action === "reply" ? 45.0 : 2.0),
      }),
    });
    const data = await res.json();

    if (action === "like") {
      appState.likedPosts.add(postId);
      const countEl = document.getElementById(`like-count-${postId}`);
      if (countEl) countEl.textContent = data.post_likes;
      showToast("❤️ Liked! Model confidence updated in real-time.");
      // Re-render card to show filled heart
      const card = document.getElementById(`post-card-${postId}`);
      if (card) {
        const heartBtn = card.querySelector("button");
        if (heartBtn) heartBtn.classList.add("text-rose-500", "font-bold");
      }
    } else if (action === "reply") {
      showToast("💬 Reply logged! Positive discussion weight applied.");
    } else if (action === "skip") {
      showToast("✕ Skipped! Post removed via negative skip feedback.");
      const card = document.getElementById(`post-card-${postId}`);
      if (card) {
        card.style.opacity = "0";
        card.style.transform = "scale(0.95)";
        setTimeout(() => card.remove(), 250);
      }
    }

    // Update user interaction count in UI
    const interEl = document.getElementById("user-interactions-count");
    if (interEl) interEl.textContent = data.user_total_interactions;
  } catch (err) {
    console.error("Interaction logging error:", err);
  }
}

// ============================================================
// PIPELINE FUNNEL INSPECTOR TELEMETRY
// ============================================================

function updateFunnelInspector(stats) {
  if (!stats) return;

  const latEl = document.getElementById("funnel-latency");
  if (latEl) latEl.textContent = `${stats.latency_ms} ms`;

  const retEl = document.getElementById("funnel-retrieved-count");
  if (retEl) retEl.textContent = stats.retrieved_count;

  const hydEl = document.getElementById("funnel-hydrated-count");
  if (hydEl) hydEl.textContent = stats.hydrated_count;

  const filEl = document.getElementById("funnel-filtered-count");
  if (filEl) filEl.textContent = stats.filtered_count;

  const feedEl = document.getElementById("funnel-feed-count");
  if (feedEl) feedEl.textContent = stats.feed_count;

  // Animate progress widths
  const maxRet = Math.max(1, stats.corpus_count);
  const wRet = Math.min(100, Math.round((stats.retrieved_count / 400) * 100));
  const wFil = Math.min(100, Math.round((stats.filtered_count / 400) * 100));

  const barRet = document.getElementById("bar-retrieved");
  if (barRet) barRet.style.width = `${wRet}%`;

  const barHyd = document.getElementById("bar-hydrated");
  if (barHyd) barHyd.style.width = `${wRet}%`;

  const barFil = document.getElementById("bar-filtered");
  if (barFil) barFil.style.width = `${wFil}%`;

  // Filter breakdown
  const fb = stats.filter_breakdown || {};
  const dropEl = document.getElementById("funnel-filter-drops");
  if (dropEl) {
    dropEl.textContent = `Self: -${fb.dropped_self || 0} • Block: -${fb.dropped_blocked || 0} • Seen: -${fb.dropped_seen || 0} • Spam: -${fb.dropped_author_spam || 0} • Bots: -${fb.dropped_shilling || 0}`;
  }
}

// ============================================================
// PERSONA SWITCHING
// ============================================================

function updateUserProfileUI() {
  const user = appState.users.find(u => u.user_id === appState.currentUserId);
  if (!user) return;

  const nameEl = document.getElementById("user-display-name");
  const handleEl = document.getElementById("user-handle");
  const avatarEl = document.getElementById("user-avatar");
  const deptEl = document.getElementById("user-dept");
  const personaEl = document.getElementById("user-persona");
  const interEl = document.getElementById("user-interactions-count");
  const followEl = document.getElementById("user-following-count");
  const folwerEl = document.getElementById("user-followers-count");
  const badgeEl = document.getElementById("cold-start-badge");
  const tagsEl = document.getElementById("user-tags");

  if (nameEl) nameEl.textContent = user.username;
  if (handleEl) handleEl.textContent = `@${user.username.toLowerCase()}`;
  if (avatarEl) avatarEl.textContent = user.username.substring(0, 2).toUpperCase();
  if (deptEl) deptEl.textContent = user.department;
  if (personaEl) personaEl.textContent = user.persona;
  if (interEl) interEl.textContent = user.interaction_count;
  if (followEl) followEl.textContent = user.following_count;
  if (folwerEl) folwerEl.textContent = user.followers_count;

  if (badgeEl) {
    if (user.is_cold_start) badgeEl.classList.remove("hidden");
    else badgeEl.classList.add("hidden");
  }

  if (tagsEl) {
    tagsEl.innerHTML = (user.preferred_tags || [])
      .map(t => `<span class="bg-[#161f30] text-sky-400 text-[10px] px-2 py-0.5 rounded-full border border-[#243046]">${t}</span>`)
      .join("");
  }

  const select = document.getElementById("user-dropdown");
  if (select) select.value = user.user_id;
}

function onUserSelectChange(newId) {
  appState.currentUserId = parseInt(newId);
  updateUserProfileUI();
  loadFeed();
  checkAttackStatus();
  showToast(`Switched active user to #${appState.currentUserId}`);
}

function switchQuickPersona(key) {
  let targetUser = null;
  if (key === "cs_warm") {
    targetUser = appState.users.find(u => !u.is_cold_start && u.persona === "CS_Undergrad");
  } else if (key === "bio_warm") {
    targetUser = appState.users.find(u => !u.is_cold_start && u.persona === "Bio_Researcher");
  } else if (key === "freshman_cold") {
    targetUser = appState.users.find(u => u.is_cold_start && u.persona === "Freshman");
  } else if (key === "club_publisher") {
    targetUser = appState.users.find(u => u.persona === "Campus_Club");
  }

  if (targetUser) {
    onUserSelectChange(targetUser.user_id);
  }
}

// ============================================================
// LIVE HYPERPARAMETER SLIDERS
// ============================================================

function onSliderChange() {
  const mmr = parseFloat(document.getElementById("slider-mmr").value);
  const wl = parseFloat(document.getElementById("slider-wlike").value);
  const wr = parseFloat(document.getElementById("slider-wreply").value);
  const ws = parseFloat(document.getElementById("slider-wskip").value);

  appState.sliders.mmr_lambda = mmr;
  appState.sliders.w_like = wl;
  appState.sliders.w_reply = wr;
  appState.sliders.w_skip = ws;

  document.getElementById("val-mmr").textContent = mmr.toFixed(2);
  document.getElementById("val-wlike").textContent = wl.toFixed(2);
  document.getElementById("val-wreply").textContent = wr.toFixed(2);
  document.getElementById("val-wskip").textContent = ws.toFixed(2);

  // Debounced feed reload
  clearTimeout(appState.debounceTimer);
  appState.debounceTimer = setTimeout(() => {
    loadFeed();
  }, 200);
}

function resetSliders() {
  document.getElementById("slider-mmr").value = 0.70;
  document.getElementById("slider-wlike").value = 0.60;
  document.getElementById("slider-wreply").value = 1.20;
  document.getElementById("slider-wskip").value = 0.40;
  onSliderChange();
  showToast("Sliders reset to defaults.");
}

// ============================================================
// SHILLING ATTACK SANDBOX
// ============================================================

async function launchAttack() {
  const targetId = parseInt(document.getElementById("sandbox-target-post-select").value);
  const attackType = document.getElementById("sandbox-attack-type").value;

  try {
    const res = await fetch(`${API_BASE}/attack/inject`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_post_id: targetId,
        attack_type: attackType,
      }),
    });
    const data = await res.json();

    showToast(`🚀 Like-Bomb Injected! Injected 25 bot profiles targeting Post #${targetId}`);
    
    // Show attack alert banner
    const alertEl = document.getElementById("attack-alert-banner");
    if (alertEl) alertEl.classList.remove("hidden");

    checkAttackStatus();
    loadFeed();
  } catch (err) {
    console.error("Attack injection error:", err);
  }
}

async function resetAttack() {
  try {
    const res = await fetch(`${API_BASE}/attack/reset`, { method: "POST" });
    const data = await res.json();

    showToast("🔄 Attack cleared! All bot accounts excised and organic rank restored.");

    const alertEl = document.getElementById("attack-alert-banner");
    if (alertEl) alertEl.classList.add("hidden");

    checkAttackStatus();
    loadFeed();
  } catch (err) {
    console.error("Attack reset error:", err);
  }
}

async function checkAttackStatus() {
  try {
    const res = await fetch(`${API_BASE}/attack/status?user_id=${appState.currentUserId}`);
    const data = await res.json();

    const statusEl = document.getElementById("sandbox-status-text");
    const rankEl = document.getElementById("sandbox-rank-indicator");
    const anomalyEl = document.getElementById("sandbox-anomaly-meter");
    const defStateEl = document.getElementById("sandbox-defense-state");

    if (data.active_attack) {
      if (statusEl) statusEl.innerHTML = `<span class="text-rose-400 font-bold">⚠️ ATTACK ACTIVE (${data.active_attack.toUpperCase()})</span>`;
      if (rankEl) {
        if (data.target_post_rank_unprotected) {
          rankEl.innerHTML = `<span class="text-rose-400 font-bold">Rank #${data.target_post_rank_unprotected} in Feed (Inflated!)</span>`;
        } else {
          rankEl.innerHTML = `<span class="text-amber-400">Post #${data.target_post_id} Sourced</span>`;
        }
      }
      if (anomalyEl) {
        anomalyEl.innerHTML = `<span class="text-rose-400 font-bold">${(data.anomaly_score * 100).toFixed(1)}% (FLAGGED BOTNET)</span>`;
      }
    } else {
      if (statusEl) statusEl.textContent = "Clean / Organic Feed";
      if (rankEl) rankEl.textContent = "Organic Rank (> #35)";
      if (anomalyEl) anomalyEl.textContent = "0.0% (Normal)";
    }

    if (defStateEl) {
      defStateEl.textContent = appState.defenseActive ? "ACTIVE (Anomaly Filter On)" : "INACTIVE";
      defStateEl.className = appState.defenseActive ? "font-bold text-emerald-400" : "font-semibold text-slate-400";
    }
  } catch (err) {
    console.error("Attack status error:", err);
  }
}

function toggleDefenseFromBanner() {
  appState.defenseActive = true;
  const toggle = document.getElementById("defense-toggle-header");
  if (toggle) toggle.checked = true;
  loadFeed();
  checkAttackStatus();
  showToast("🛡️ Shilling Defense Filter ACTIVATED! Target post excised from feed.");
}

// ============================================================
// ACADEMIC BENCHMARK TAB
// ============================================================

async function loadBenchmarks() {
  try {
    const res = await fetch(`${API_BASE}/benchmark/summary`);
    const data = await res.json();

    // 1. Table 1: Main Benchmarks
    const tMain = document.getElementById("benchmark-table-main");
    if (tMain && data.main_benchmarks) {
      tMain.innerHTML = data.main_benchmarks.map(m => `
        <tr class="hover:bg-[#161f30]/60 transition ${m.model.includes('Hybrid') ? 'bg-sky-500/10 font-bold text-sky-300' : ''}">
          <td class="p-2.5 font-sans">${m.model}</td>
          <td class="p-2.5">${m['p@10'].toFixed(4)}</td>
          <td class="p-2.5">${m['r@10'].toFixed(4)}</td>
          <td class="p-2.5 text-emerald-400 font-bold">${m['ndcg@10'].toFixed(4)}</td>
          <td class="p-2.5">${m.coverage_pct.toFixed(2)}%</td>
          <td class="p-2.5">${m.ild_diversity.toFixed(4)}</td>
          <td class="p-2.5">${m.rmse !== null ? m.rmse.toFixed(4) : 'N/A'}</td>
        </tr>
      `).join("");
    }

    // 2. Table 2: Cold-Start Cohort
    const tCold = document.getElementById("benchmark-table-cold");
    if (tCold && data.cold_start_benchmarks) {
      tCold.innerHTML = data.cold_start_benchmarks.map(m => `
        <tr class="hover:bg-[#161f30]/60 transition ${m.model.includes('Hybrid') ? 'bg-amber-500/10 font-bold text-amber-300' : ''}">
          <td class="p-2.5 font-sans">${m.model}</td>
          <td class="p-2.5">${m['p@10'].toFixed(4)}</td>
          <td class="p-2.5">${m['r@10'].toFixed(4)}</td>
          <td class="p-2.5 text-amber-400 font-bold">${m['ndcg@10'].toFixed(4)}</td>
          <td class="p-2.5">${m.coverage_pct.toFixed(2)}%</td>
          <td class="p-2.5">${m.ild_diversity.toFixed(4)}</td>
        </tr>
      `).join("");
    }

    // 3. Table 3: Ablations
    const tAbl = document.getElementById("benchmark-table-ablations");
    if (tAbl && data.ablations) {
      tAbl.innerHTML = data.ablations.map(m => `
        <tr class="hover:bg-[#161f30]/60 transition">
          <td class="p-2.5 font-sans">${m.model}</td>
          <td class="p-2.5">${m['p@10'].toFixed(4)}</td>
          <td class="p-2.5">${m['r@10'].toFixed(4)}</td>
          <td class="p-2.5 text-purple-400 font-bold">${m['ndcg@10'].toFixed(4)}</td>
          <td class="p-2.5">${m.coverage_pct.toFixed(2)}%</td>
          <td class="p-2.5">${m.ild_diversity.toFixed(4)}</td>
        </tr>
      `).join("");
    }
  } catch (err) {
    console.error("Benchmark load error:", err);
  }
}

// ============================================================
// NAVIGATION & UTILS
// ============================================================

function switchTab(tabId) {
  appState.activeTab = tabId;

  // Toggle tab contents
  document.getElementById("tab-feed-container").classList.add("hidden");
  document.getElementById("tab-benchmarks-container").classList.add("hidden");
  document.getElementById("tab-security-container").classList.add("hidden");

  // Reset tab button styles
  const btnFeed = document.getElementById("nav-feed");
  const btnBench = document.getElementById("nav-benchmarks");
  const btnSec = document.getElementById("nav-security");

  [btnFeed, btnBench, btnSec].forEach(b => {
    b.className = "w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl hover:bg-[#161f30] text-slate-300 text-left transition";
  });

  if (tabId === "feed") {
    document.getElementById("tab-feed-container").classList.remove("hidden");
    btnFeed.className = "w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl bg-sky-500/10 text-sky-400 border border-sky-500/20 text-left transition";
  } else if (tabId === "benchmarks") {
    document.getElementById("tab-benchmarks-container").classList.remove("hidden");
    btnBench.className = "w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/20 text-left transition";
    loadBenchmarks();
  } else if (tabId === "security") {
    document.getElementById("tab-security-container").classList.remove("hidden");
    btnSec.className = "w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl bg-rose-500/10 text-rose-400 border border-rose-500/20 text-left transition";
    checkAttackStatus();
  }
}

function filterFeedByTag(tag) {
  if (!tag) return;
  const cleanTag = tag.trim().toLowerCase();
  const filtered = appState.activeFeed.filter(p => {
    return (
      p.content.toLowerCase().includes(cleanTag) ||
      (p.tags || []).some(t => t.toLowerCase().includes(cleanTag))
    );
  });
  if (filtered.length > 0) {
    renderFeed(filtered);
    showToast(`Filtered feed for "${cleanTag}" (${filtered.length} matches)`);
  } else {
    showToast(`No posts found for "${cleanTag}" in current feed.`);
  }
}

// ============================================================
// CASE-BASED CAMPUS TOPIC SEARCH (Lecture L29)
// ============================================================

async function handleCaseSearch(query) {
  if (!query || !query.trim()) {
    showToast("Please enter a campus topic or query to search.");
    return;
  }
  const cleanQ = query.trim();
  const feedList = document.getElementById("feed-list");
  if (feedList) {
    feedList.innerHTML = `
      <div class="text-center py-12 text-slate-500 text-sm">
        <div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500 mb-2"></div>
        <p>Executing Case-Based TF-IDF query match across 3,500 campus posts...</p>
      </div>
    `;
  }

  try {
    const res = await fetch(`${API_BASE}/search?q=${encodeURIComponent(cleanQ)}&user_id=${appState.currentUserId}&top_k=15`);
    const data = await res.json();
    appState.activeFeed = data.feed || [];
    renderFeed(appState.activeFeed);

    const clearBtn = document.getElementById("search-clear-btn");
    if (clearBtn) clearBtn.classList.remove("hidden");

    showToast(`Found ${data.results_count || appState.activeFeed.length} case-based matches for "${cleanQ}"`);
  } catch (err) {
    console.error("Search error:", err);
    showToast("Error searching campus posts.");
  }
}

function clearSearch() {
  const input = document.getElementById("campus-search-input");
  if (input) input.value = "";
  const clearBtn = document.getElementById("search-clear-btn");
  if (clearBtn) clearBtn.classList.add("hidden");
  loadFeed();
  showToast("Restored personalized 'For You' feed.");
}

function showToast(message) {
  const toast = document.getElementById("toast");
  if (!toast) return;

  toast.textContent = message;
  toast.classList.remove("translate-y-20", "opacity-0");
  toast.classList.add("translate-y-0", "opacity-100");

  setTimeout(() => {
    toast.classList.remove("translate-y-0", "opacity-100");
    toast.classList.add("translate-y-20", "opacity-0");
  }, 2800);
}
