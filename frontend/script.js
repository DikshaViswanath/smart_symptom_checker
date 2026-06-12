/**
 * Smart Symptom Checker — AI Healthcare Chatbot Frontend
 * Connects to Flask backend at POST http://127.0.0.1:5000/predict
 */

const API_BASE = "http://127.0.0.1:5000";
const API_PREDICT_URL = `${API_BASE}/predict`;
const API_HEALTH_URL = `${API_BASE}/`;

/**
 * Safe synonym map for session memory & extraction only.
 * Maps explicitly equivalent phrases — never infers unrelated symptoms.
 */
const SAFE_DISPLAY_SYNONYMS = {
  "high fever": "fever",
  "mild fever": "fever",
  "high temperature": "fever",
  "pain in chest": "chest pain",
  "pain in my chest": "chest pain",
  "chest discomfort": "chest pain",
  "head ache": "headache",
  "head pain": "headache",
  "body pain": "body ache",
  "muscle pain": "body ache",
  "dry cough": "cough",
  "wet cough": "cough",
  "runny nose": "cold",
  sniffles: "cold",
  "common cold": "cold",
  tiredness: "fatigue",
  exhausted: "fatigue",
  breathlessness: "shortness of breath",
  "breathing difficulty": "shortness of breath",
  "difficulty breathing": "shortness of breath",
  vertigo: "dizziness",
  lightheaded: "dizziness",
};

/** Applied only when sending to Flask — not stored in chat memory. */
const API_MODEL_ALIASES = {
  ...SAFE_DISPLAY_SYNONYMS,
  cold: "runny nose",
  "stuffy nose": "nasal congestion",
  "blocked nose": "nasal congestion",
  "chronic cough": "cough",
};

const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const symptomInput = document.getElementById("symptomInput");
const sendBtn = document.getElementById("sendBtn");
const sendBtnText = document.getElementById("sendBtnText");
const sendBtnSpinner = document.getElementById("sendBtnSpinner");
const errorAlert = document.getElementById("errorAlert");
const errorMessage = document.getElementById("errorMessage");
const dismissError = document.getElementById("dismissError");
const micBtn = document.getElementById("micBtn");
const micIcon = document.getElementById("micIcon");
const voiceStatus = document.getElementById("voiceStatus");
const voiceStatusText = document.getElementById("voiceStatusText");
const statusBadge = document.getElementById("statusBadge");
const statusLabel = document.getElementById("statusLabel");

let isProcessing = false;
let conversationBusy = false;
let backendReady = false;
let modelLoaded = false;

/* ============================================
   Voice Input Module (Speech Recognition API)
   ============================================ */
const VoiceInput = (() => {
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;

  const STATUS = {
    LISTENING: "Listening…",
    DETECTED: "Speech detected",
    FAILED: "Recognition failed",
    STOPPED: "Voice input stopped",
    IDLE: "",
  };

  let recognition = null;
  let isListening = false;
  let isEnabled = true;
  let statusHideTimer = null;
  let inputSnapshot = "";
  let elements = {};

  function isSupported() {
    return Boolean(SpeechRecognition);
  }

  function setStatus(message, type) {
    if (!elements.statusEl || !elements.statusTextEl) return;

    clearTimeout(statusHideTimer);

    if (!message) {
      elements.statusEl.classList.add("d-none");
      elements.statusEl.classList.remove("listening", "detected", "failed", "unsupported");
      return;
    }

    elements.statusEl.classList.remove("d-none", "listening", "detected", "failed", "unsupported");
    if (type) {
      elements.statusEl.classList.add(type);
    }

    const iconClass =
      type === "detected"
        ? "bi-check-circle-fill"
        : type === "failed"
          ? "bi-exclamation-circle-fill"
          : "bi-mic-fill";

    const iconWrap = elements.statusEl.querySelector(".voice-status-icon");
    if (iconWrap) {
      iconWrap.innerHTML = `<i class="bi ${iconClass}"></i>`;
    }

    elements.statusTextEl.textContent = message;
  }

  function scheduleStatusHide(delay = 2500) {
    clearTimeout(statusHideTimer);
    statusHideTimer = setTimeout(() => {
      if (!isListening) {
        setStatus("", null);
      }
    }, delay);
  }

  function setListeningUI(active) {
    isListening = active;

    if (elements.micBtn) {
      elements.micBtn.classList.toggle("listening", active);
      elements.micBtn.setAttribute("aria-label", active ? "Stop voice input" : "Start voice input");
      elements.micBtn.setAttribute("aria-pressed", String(active));
    }

    if (elements.micIcon) {
      elements.micIcon.className = active ? "bi bi-mic-mute-fill mic-icon" : "bi bi-mic-fill mic-icon";
    }

    if (elements.inputEl) {
      elements.inputEl.classList.toggle("voice-active", active);
    }
  }

  function getErrorMessage(errorCode) {
    const messages = {
      "no-speech": "No speech was detected. Please try again.",
      "audio-capture": "Microphone not found. Check your device settings.",
      "not-allowed": "Microphone permission denied. Allow access in browser settings.",
      "network": "Network error during speech recognition.",
      "aborted": "Voice input was cancelled.",
      "language-not-supported": "Language not supported for recognition.",
      "service-not-allowed": "Speech recognition service is not allowed.",
    };
    return messages[errorCode] || "Could not recognize speech. Please try again.";
  }

  /** Full transcript accumulated for the current mic session (no parsing/filtering). */
  let sessionTranscript = "";

  function setInputDisplayText(text) {
    if (!elements.inputEl) return;
    elements.inputEl.value = text;
    elements.inputEl.dispatchEvent(new Event("input", { bubbles: true }));
    elements.inputEl.focus();
  }

  function buildDisplayTranscript(event) {
    let spoken = "";
    for (let i = 0; i < event.results.length; i += 1) {
      spoken += event.results[i][0].transcript;
    }
    return spoken.trim();
  }

  function stop() {
    if (recognition && isListening) {
      try {
        recognition.stop();
      } catch {
        /* recognition may already be stopped */
      }
    }
    setListeningUI(false);
  }

  function start() {
    if (!isSupported() || !recognition || !isEnabled || isListening) return;

    hideErrorIfAvailable();
    setStatus(STATUS.LISTENING, "listening");
    setListeningUI(true);

    try {
      recognition.start();
    } catch (error) {
      setListeningUI(false);
      if (error.name === "InvalidStateError") {
        try {
          recognition.stop();
          recognition.start();
        } catch {
          setStatus(`${STATUS.FAILED}: Already running.`, "failed");
          scheduleStatusHide(3000);
        }
      } else {
        setStatus(`${STATUS.FAILED}: ${error.message}`, "failed");
        scheduleStatusHide(3000);
      }
    }
  }

  function toggle() {
    if (!isSupported()) {
      showUnsupportedMessage();
      return;
    }
    if (!isEnabled) return;

    if (isListening) {
      stop();
      setStatus(STATUS.STOPPED, "detected");
      scheduleStatusHide();
    } else {
      start();
    }
  }

  function showUnsupportedMessage() {
    setStatus(
      "Voice input is not supported in this browser. Please use Chrome or Edge, or type your symptoms.",
      "unsupported"
    );
    if (typeof showError === "function") {
      showError(
        "Speech recognition is not supported in your browser. Try Google Chrome, Microsoft Edge, or Safari."
      );
    }
    scheduleStatusHide(6000);
  }

  function hideErrorIfAvailable() {
    if (typeof hideError === "function") {
      hideError();
    }
  }

  function bindRecognitionEvents() {
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      inputSnapshot = elements.inputEl ? elements.inputEl.value.trim() : "";
      sessionTranscript = "";
      setStatus(STATUS.LISTENING, "listening");
      setListeningUI(true);
    };

    recognition.onresult = (event) => {
      sessionTranscript = buildDisplayTranscript(event);
      const display = inputSnapshot
        ? `${inputSnapshot} ${sessionTranscript}`.trim()
        : sessionTranscript;
      setInputDisplayText(display);

      const hasFinal = Array.from(event.results).some((r) => r.isFinal);
      setStatus(hasFinal ? STATUS.DETECTED : STATUS.LISTENING, hasFinal ? "detected" : "listening");
    };

    recognition.onerror = (event) => {
      setListeningUI(false);

      if (event.error === "aborted") {
        setStatus(STATUS.STOPPED, "detected");
        scheduleStatusHide();
        return;
      }

      const detail = getErrorMessage(event.error);
      setStatus(`${STATUS.FAILED}: ${detail}`, "failed");

      if (event.error === "not-allowed" || event.error === "audio-capture") {
        if (typeof showError === "function") {
          showError(detail);
        }
      }

      scheduleStatusHide(4000);
    };

    recognition.onend = () => {
      setListeningUI(false);

      if (sessionTranscript.trim()) {
        const display = inputSnapshot
          ? `${inputSnapshot} ${sessionTranscript}`.trim()
          : sessionTranscript.trim();
        setInputDisplayText(display);
      }

      if (elements.inputEl && elements.inputEl.value.trim()) {
        setStatus(STATUS.DETECTED, "detected");
        scheduleStatusHide();
      } else if (!elements.statusEl.classList.contains("failed")) {
        scheduleStatusHide(1500);
      }
    };
  }

  function setEnabled(enabled) {
    isEnabled = enabled;

    if (elements.micBtn) {
      elements.micBtn.disabled = !enabled;
    }

    if (!enabled) {
      stop();
    }
  }

  function init(config) {
    elements = {
      inputEl: config.inputEl,
      micBtn: config.micBtn,
      micIcon: config.micIcon,
      statusEl: config.statusEl,
      statusTextEl: config.statusTextEl,
    };

    if (!isSupported()) {
      if (elements.micBtn) {
        elements.micBtn.classList.add("mic-unsupported");
        elements.micBtn.title = "Voice input not supported — click for details";
        elements.micBtn.addEventListener("click", showUnsupportedMessage);
      }
      return false;
    }

    recognition = new SpeechRecognition();
    bindRecognitionEvents();

    elements.micBtn.addEventListener("click", toggle);

    return true;
  }

  return {
    init,
    isSupported,
    start,
    stop,
    toggle,
    setEnabled,
    isListening: () => isListening,
  };
})();

/* ============================================
   Symptom NLP — extract medical terms only
   ============================================ */
const SymptomNLP = (() => {
  const KNOWN_SYMPTOMS = [
    "common cold", "runny nose", "stuffy nose", "blocked nose", "nasal congestion",
    "sniffles", "cold", "high temperature", "body pain", "muscle pain", "head ache",
    "chest pain", "chest discomfort", "chest tightness", "chest pressure",
    "breathing difficulty", "difficulty breathing", "shortness of breath",
    "dry cough", "wet cough", "sore throat", "loss of appetite", "low energy",
    "fever", "cough", "headache", "migraine", "nausea", "dizziness", "vomiting",
    "fatigue", "tiredness", "weakness", "body ache", "aches", "soreness",
    "chills", "sweating", "vertigo", "lightheaded", "temperature", "congestion",
    "sneezing", "wheezing", "rash", "itching", "heartburn", "constipation",
    "abdominal pain", "back pain", "joint pain", "sore muscles", "coughing",
    "exhausted", "pain behind eyes", "blurred vision", "burning urination",
    "frequent urination", "cloudy urine", "pelvic pain", "facial pain",
  ];

  const FILLER_PHRASES = [
    "as mentioned earlier", "as i said before", "as i mentioned before",
    "like i said before", "i said before", "i mentioned before",
    "as mentioned before", "as said earlier", "like i mentioned",
    "i also have", "i've also got", "i have also", "also have",
    "in addition", "additionally", "plus i have", "and also",
    "by the way", "oh and", "one more thing",
  ];

  const LEADING_PATTERNS = [
    /^i\s+(also\s+)?(have|am\s+having|feel|am\s+feeling|got|get|experience|am\s+experiencing)\s+/i,
    /^i\s+(also\s+)?(think\s+i\s+have|believe\s+i\s+have)\s+/i,
    /^my\s+symptoms?\s+(are|is|include|also\s+include)\s+/i,
    /^symptoms?\s*:\s*/i,
    /^currently\s+(i\s+)?(have|feel)\s+/i,
    /^and\s+i\s+(also\s+)?(have|got)\s+/i,
    /^plus\s+(i\s+)?(have|got)\s+/i,
  ];

  const STOP_WORDS = new Set([
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "no", "nope", "nah",
    "none", "nothing", "not", "really", "too", "also", "just", "only",
    "maybe", "sometimes", "little", "bit", "quite", "very", "still",
    "well", "um", "uh", "oh", "ah", "like", "so", "now", "then",
    "both", "all", "some", "any", "the", "a", "an", "my", "me", "i",
    "do", "does", "did", "am", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "having", "get", "got", "getting",
    "mentioned", "earlier", "before", "said", "told", "again",
    "think", "feel", "feeling", "experiencing", "experience",
  ]);

  const PHRASE_PATTERNS = [
    ["pain in my chest", "chest pain"],
    ["pain in chest", "chest pain"],
    ["chest pain", "chest pain"],
    ["my chest is paining", "chest pain"],
    ["chest is paining", "chest pain"],
    ["my chest hurts", "chest pain"],
    ["chest hurts", "chest pain"],
    ["pain near chest", "chest pain"],
    ["chest hurts", "chest pain"],
    ["chest discomfort", "chest pain",],
    ["body pain", "body ache"],
    ["high temperature", "fever"],
    ["high fever", "fever"],
  ];

  const sortedTerms = [...KNOWN_SYMPTOMS].sort((a, b) => b.length - a.length);

  const allowedCanonical = new Set(
    KNOWN_SYMPTOMS.map((s) => canonicalizeForMemory(s))
  );

  function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function canonicalizeForMemory(symptom) {
    const k = symptom.trim().toLowerCase();
    return SAFE_DISPLAY_SYNONYMS[k] || k;
  }

  /** @deprecated use canonicalizeForMemory */
  function normalizeKey(symptom) {
    return canonicalizeForMemory(symptom);
  }

  function symptomsEquivalent(a, b) {
    return canonicalizeForMemory(a) === canonicalizeForMemory(b);
  }

  function symptomMatches(a, b) {
    return symptomsEquivalent(a, b);
  }

  function isAllowedSymptom(term) {
    return allowedCanonical.has(canonicalizeForMemory(term));
  }

  function cleanText(input) {
    let text = input.trim().toLowerCase();
    if (!text) return "";

    FILLER_PHRASES.sort((a, b) => b.length - a.length).forEach((phrase) => {
      text = text.replace(new RegExp(escapeRegex(phrase), "gi"), " ");
    });

    LEADING_PATTERNS.forEach((pattern) => {
      text = text.replace(pattern, "");
    });

    text = text
      .replace(/\b(too|also|just|even|still|already)\b/gi, " ")
      .replace(/\s+/g, " ")
      .trim();

    return text;
  }

  function extractSymptoms(input) {
    if (!input || !input.trim()) return [];

    let text = cleanText(input);
    if (!text) return [];

    const found = [];
    const seen = new Set();

    function addSymptom(term) {
      const canonical = canonicalizeForMemory(term);
      if (!canonical || !allowedCanonical.has(canonical) || seen.has(canonical)) return;
      seen.add(canonical);
      found.push(canonical);
    }

    PHRASE_PATTERNS.forEach(([phrase, canonical]) => {
      if (text.includes(phrase)) {
        addSymptom(canonical);
        text = text.replace(new RegExp(escapeRegex(phrase), "gi"), " ");
      }
    });

    sortedTerms.forEach((term) => {
      const re = new RegExp(`\\b${escapeRegex(term)}\\b`, "gi");
      if (re.test(text)) {
        addSymptom(term);
        text = text.replace(re, " ");
      }
    });

    const parts = text
      .split(/[,;]+|\band\b|\bor\b|\bwith\b|\bplus\b/i)
      .map((s) => s.trim())
      .filter((s) => s.length > 1 && !STOP_WORDS.has(s));

    parts.forEach((part) => {
      const cleaned = part
        .replace(/^(a|an|the|some|mild|severe|bad|slight)\s+/i, "")
        .trim();
      if (!cleaned || STOP_WORDS.has(cleaned)) return;

      const direct = sortedTerms.find((t) => symptomsEquivalent(t, cleaned));
      if (direct) {
        addSymptom(direct);
      } else if (isAllowedSymptom(cleaned)) {
        addSymptom(cleaned);
      }
    });

    return found;
  }

  function isNegativeResponse(input) {
    const text = input.trim().toLowerCase();
    return /^(no|nope|nah|none|nothing|not really|don't|do not|no i don't|not at all|negative)\b/i.test(
      text
    );
  }

  function isAffirmativeOnly(input) {
    const text = input.trim().toLowerCase();
    if (extractSymptoms(input).length > 0) return false;
    return /^(yes|yeah|yep|yup|sure|i do|i have|sometimes|a little|i think so|correct|right)\.?$/i.test(
      text
    );
  }

  return {
    extractSymptoms,
    isNegativeResponse,
    isAffirmativeOnly,
    symptomMatches,
    symptomsEquivalent,
    canonicalizeForMemory,
    normalizeKey,
  };
})();

/* ============================================
   Conversational State & Follow-Up Engine
   ============================================ */
const ConversationManager = (() => {
  const State = {
    IDLE: "idle",
    FOLLOW_UP: "follow_up",
    ANALYZING: "analyzing",
  };

  /** Symptom-specific follow-up rules (asked one at a time, in order). */
  const FOLLOW_UP_RULES = [
    {
      id: "cold",
      triggers: ["cold", "common cold", "runny nose", "stuffy nose", "sniffles", "blocked nose"],
      question:
        "Do you also have <strong>cough</strong> or <strong>fever</strong>?",
      hints: ["cough", "fever", "high temperature", "sore throat", "chills"],
    },
    {
      id: "fever",
      triggers: ["fever", "high temperature", "temperature", "chills"],
      question:
        "Do you also have <strong>cough</strong> or <strong>body pain</strong>?",
      hints: ["cough", "body pain", "body ache", "muscle pain", "aches", "soreness"],
    },
    {
      id: "body_pain",
      triggers: ["body pain", "body ache", "muscle pain", "aches", "soreness", "sore muscles"],
      question:
        "Are you also experiencing <strong>fatigue</strong> or <strong>headache</strong>?",
      hints: ["fatigue", "tiredness", "headache", "head ache", "weakness", "exhausted"],
    },
    {
      id: "cough",
      triggers: ["cough", "coughing", "dry cough", "wet cough"],
      question:
        "Do you also have <strong>fever</strong> or <strong>sore throat</strong>?",
      hints: ["fever", "high temperature", "sore throat", "chills"],
    },
    {
      id: "headache",
      triggers: ["headache", "head ache", "migraine"],
      question:
        "Do you also feel <strong>nausea</strong> or <strong>dizziness</strong>?",
      hints: ["nausea", "dizziness", "vomiting", "vertigo", "lightheaded"],
    },
    {
      id: "fatigue",
      triggers: ["fatigue", "tiredness", "exhausted", "weakness", "low energy"],
      question:
        "Do you also have <strong>fever</strong> or <strong>body pain</strong>?",
      hints: ["fever", "high temperature", "body pain", "body ache"],
    },
    {
      id: "chest_pain",
      triggers: ["chest pain", "chest discomfort", "chest tightness", "chest pressure"],
      question:
        "Do you have <strong>breathing difficulty</strong> or <strong>sweating</strong>?",
      hints: [
        "breathing difficulty",
        "breathing problem",
        "shortness of breath",
        "breathlessness",
        "sweating",
        "difficulty breathing",
      ],
    },
  ];

  /** Used when no specific rule matches — still ask before calling the backend. */
  const DEFAULT_FOLLOW_UP = {
    id: "general",
    triggers: [],
    question:
      "Are you experiencing any other symptoms, such as <strong>cough</strong>, <strong>fever</strong>, <strong>fatigue</strong>, or <strong>headache</strong>?",
    hints: [
      "cough",
      "fever",
      "fatigue",
      "headache",
      "body pain",
      "body ache",
      "nausea",
      "dizziness",
      "sore throat",
      "chills",
    ],
  };

  let state = State.IDLE;
  let collectedSymptoms = [];
  let pendingRules = [];
  let currentRule = null;
  let askedRuleIds = new Set();
  /** True only after a follow-up question was shown — user must reply before analyze. */
  let awaitingFollowUpAnswer = false;
  /** @type {{ role: "user"|"bot", text: string }[]} */
  let sessionMessages = [];

  function normalize(text) {
    return text.toLowerCase().trim();
  }

  function symptomMatchesKeyword(symptom, keyword) {
    return SymptomNLP.symptomMatches(symptom, keyword);
  }

  function recordMessage(role, text) {
    if (text && text.trim()) {
      sessionMessages.push({ role, text: text.trim() });
    }
  }

  function hasSymptom(symptom) {
    return collectedSymptoms.some((s) => SymptomNLP.symptomMatches(s, symptom));
  }

  function hasTrigger(symptoms, rule) {
    if (!rule.triggers || rule.triggers.length === 0) return false;
    return rule.triggers.some((trigger) =>
      symptoms.some((symptom) => symptomMatchesKeyword(symptom, trigger))
    );
  }

  function hasRelatedSymptoms(symptoms, rule) {
    return rule.hints.some((hint) =>
      symptoms.some((symptom) => SymptomNLP.symptomsEquivalent(symptom, hint))
    );
  }

  function buildFollowUpQueue(symptoms) {
    const matched = FOLLOW_UP_RULES.filter(
      (rule) => hasTrigger(symptoms, rule) && !hasRelatedSymptoms(symptoms, rule)
    );
    if (matched.length > 0) return matched;
    if (!hasRelatedSymptoms(symptoms, DEFAULT_FOLLOW_UP)) {
      return [DEFAULT_FOLLOW_UP];
    }
    return [];
  }

  function mergeSymptoms(newSymptoms) {
    const seen = new Set(
      collectedSymptoms.map((s) => SymptomNLP.canonicalizeForMemory(s))
    );

    newSymptoms.forEach((symptom) => {
      const canonical = SymptomNLP.canonicalizeForMemory(symptom);
      if (!canonical || seen.has(canonical)) return;
      seen.add(canonical);
      collectedSymptoms.push(canonical);
    });

    return collectedSymptoms;
  }

  function refreshFollowUpQueue() {
    const candidates = buildFollowUpQueue(collectedSymptoms).filter(
      (rule) => !askedRuleIds.has(rule.id)
    );
    pendingRules = candidates;
    currentRule = pendingRules[0] || null;
    if (currentRule) {
      state = State.FOLLOW_UP;
    } else if (state !== State.ANALYZING) {
      state = State.IDLE;
    }
  }

  function markCurrentRuleAsked() {
    if (currentRule) {
      askedRuleIds.add(currentRule.id);
    }
  }

  function getNewlyAddedSymptoms(extracted) {
    return extracted.filter((s) => !hasSymptom(s));
  }

  /**
   * Parse follow-up reply — extract only medical symptoms, handle yes/no.
   * @returns {{ added: string[], needsClarification: boolean, isNegative: boolean }}
   */
  function parseFollowUpResponse(input) {
    const extracted = SymptomNLP.extractSymptoms(input);
    const isNegative = SymptomNLP.isNegativeResponse(input) && extracted.length === 0;

    if (extracted.length > 0) {
      return { added: extracted, needsClarification: false, isNegative: false };
    }

    if (isNegative) {
      return { added: [], needsClarification: false, isNegative: true };
    }

    if (SymptomNLP.isAffirmativeOnly(input)) {
      return { added: [], needsClarification: true, isNegative: false };
    }

    return { added: [], needsClarification: false, isNegative: false };
  }

  function getCurrentQuestion() {
    return currentRule ? currentRule.question : null;
  }

  function reset() {
    state = State.IDLE;
    collectedSymptoms = [];
    pendingRules = [];
    currentRule = null;
    askedRuleIds = new Set();
    awaitingFollowUpAnswer = false;
    sessionMessages = [];
  }

  function markQuestionPresented() {
    if (currentRule) {
      state = State.FOLLOW_UP;
      awaitingFollowUpAnswer = true;
    }
  }

  function isAwaitingFollowUp() {
    return (
      state === State.FOLLOW_UP &&
      currentRule !== null &&
      awaitingFollowUpAnswer
    );
  }

  function hasActiveMemory() {
    return collectedSymptoms.length > 0;
  }

  function isAnalyzing() {
    return state === State.ANALYZING;
  }

  function getCollectedSymptoms() {
    return [...collectedSymptoms];
  }

  function getSymptomSummaryHtml() {
    return collectedSymptoms.map((s) => `<strong>${escapeHtml(s)}</strong>`).join(", ");
  }

  function beginAnalyzing() {
    state = State.ANALYZING;
    currentRule = null;
    pendingRules = [];
  }

  function abortAnalyzing() {
    if (state === State.ANALYZING) {
      state = State.IDLE;
    }
  }

  /**
   * Start intake — merges into existing session memory (never drops earlier symptoms).
   * @param {string[]} symptoms
   */
  function beginNewSession(symptoms) {
    reset();
    return startSession(symptoms);
  }

  function startSession(symptoms) {
    const newlyAdded = getNewlyAddedSymptoms(symptoms);
    mergeSymptoms(symptoms);
    refreshFollowUpQueue();
    awaitingFollowUpAnswer = false;

    return {
      needsFollowUp: Boolean(currentRule),
      question: currentRule ? currentRule.question : null,
      symptoms: getCollectedSymptoms(),
      totalQuestions: pendingRules.length,
      newlyAdded,
    };
  }

  /**
   * Append symptoms during an active session (e.g. "I also have fever").
   * Never replaces existing memory — only merges new unique symptoms.
   * @param {string[]} symptoms
   */
  function appendSymptoms(symptoms) {
    const newlyAdded = getNewlyAddedSymptoms(symptoms);
    if (newlyAdded.length > 0) {
      mergeSymptoms(newlyAdded);
      refreshFollowUpQueue();
    }
    return {
      newlyAdded,
      symptoms: getCollectedSymptoms(),
      needsFollowUp: Boolean(currentRule),
      question: currentRule ? currentRule.question : null,
    };
  }

  /**
   * Handle one follow-up answer; merge symptoms, skip satisfied rules, avoid repeats.
   * @param {string} rawInput
   */
  function processFollowUpResponse(rawInput) {
    if (!awaitingFollowUpAnswer || !currentRule) {
      return {
        needsFollowUp: Boolean(currentRule),
        question: currentRule ? currentRule.question : null,
        symptoms: getCollectedSymptoms(),
        acknowledged: [],
      };
    }

    const { added, needsClarification, isNegative } = parseFollowUpResponse(rawInput);

    if (needsClarification) {
      const hintList = currentRule.hints
        .slice(0, 4)
        .map((h) => `<strong>${escapeHtml(h)}</strong>`)
        .join(", ");
      return {
        needsFollowUp: true,
        needsClarification: true,
        question: `Could you tell me which symptoms apply? For example: ${hintList} — or type <strong>no</strong> if none.`,
        symptoms: getCollectedSymptoms(),
        acknowledged: [],
      };
    }

    const newlyAdded = getNewlyAddedSymptoms(added);
    if (newlyAdded.length > 0) {
      mergeSymptoms(newlyAdded);
    }

    awaitingFollowUpAnswer = false;
    markCurrentRuleAsked();
    refreshFollowUpQueue();

    if (currentRule) {
      return {
        needsFollowUp: true,
        question: currentRule.question,
        symptoms: getCollectedSymptoms(),
        acknowledged: newlyAdded,
        wasNegative: isNegative && newlyAdded.length === 0,
      };
    }

    state = State.IDLE;
    return {
      needsFollowUp: false,
      symptoms: getCollectedSymptoms(),
      acknowledged: newlyAdded,
      wasNegative: isNegative && newlyAdded.length === 0,
    };
  }

  function getMemorySummaryHtml() {
    if (collectedSymptoms.length === 0) return "";
    return collectedSymptoms.map((s) => `<strong>${escapeHtml(s)}</strong>`).join(", ");
  }

  function getInputPlaceholder() {
    if (isAwaitingFollowUp()) {
      return "Reply to the question above (e.g., cough, yes, or no)...";
    }
    if (isAnalyzing()) {
      return "Analyzing your symptoms…";
    }
    return "Describe your symptoms (e.g., cold, fever, body pain)...";
  }

  return {
    reset,
    beginNewSession,
    startSession,
    appendSymptoms,
    processFollowUpResponse,
    beginAnalyzing,
    abortAnalyzing,
    isAwaitingFollowUp,
    isAnalyzing,
    hasActiveMemory,
    markQuestionPresented,
    getCollectedSymptoms,
    getSymptomSummaryHtml,
    getMemorySummaryHtml,
    recordMessage,
    getInputPlaceholder,
    getState: () => state,
  };
})();

/**
 * Promise-based delay helper.
 * @param {number} ms
 */
function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

/**
 * Show typing indicator, wait, then post a bot message.
 * @param {string} html
 * @param {number} typingDuration
 */
async function addBotMessageWithTyping(html, typingDuration = 1100) {
  const typingRow = showTypingIndicator();
  await delay(typingDuration);
  removeTypingIndicator(typingRow);
  await addBotMessage(html, { delay: 0 });
}

/**
 * Update input placeholder for current conversation phase.
 */
function updateInputPlaceholder() {
  symptomInput.placeholder = ConversationManager.getInputPlaceholder();
  symptomInput.classList.toggle(
    "input-followup-mode",
    ConversationManager.isAwaitingFollowUp()
  );
}

/**
 * Initialize the chatbot with a greeting.
 */
function initChat() {
  addBotMessage(
    "Hello! I'm your AI health assistant. I can help analyze your symptoms and suggest possible conditions along with specialist recommendations.",
    { delay: 400 }
  );

  setTimeout(() => {
    addBotMessage(
      "Tell me how you're feeling — e.g. <strong>I have cold and cough</strong>. You can add more later (<strong>I also have fever</strong>) and I'll remember everything. I ask follow-up questions <em>before</em> analyzing. Type or use the <strong><i class=\"bi bi-mic-fill\"></i> microphone</strong> — speak full sentences; I'll extract symptoms when you send.",
      { delay: 0 }
    );
  }, 900);
}

/**
 * Extract medical symptoms from user text (parsing runs only on submit).
 * @param {string} input
 * @returns {string[]}
 */
function parseSymptoms(input) {
  return SymptomNLP.extractSymptoms(input);
}

/**
 * Normalize symptom strings to canonical names expected by the backend model.
 * @param {string[]} symptoms
 * @returns {string[]}
 */
function normalizeSymptomsForApi(symptoms) {
  const seen = new Set();
  const normalized = [];

  symptoms.forEach((symptom) => {
    const key = symptom.trim().toLowerCase();
    if (!key) return;

    const canonical = API_MODEL_ALIASES[key] || key;
    if (!seen.has(canonical)) {
      seen.add(canonical);
      normalized.push(canonical);
    }
  });

  return normalized;
}

/**
 * Detect network / CORS failures from fetch errors across browsers.
 * @param {Error} error
 * @returns {boolean}
 */
function isNetworkError(error) {
  if (!error) return false;
  if (error.name === "TypeError") return true;

  const message = (error.message || "").toLowerCase();
  return (
    message.includes("fetch") ||
    message.includes("network") ||
    message.includes("failed to load") ||
    message.includes("load failed")
  );
}

/**
 * Update header status badge for backend connectivity.
 * @param {"connecting"|"online"|"degraded"|"offline"} state
 * @param {string} label
 */
function setBackendStatus(state, label) {
  if (!statusBadge || !statusLabel) return;

  statusBadge.classList.remove(
    "status-online",
    "status-offline",
    "status-degraded",
    "status-connecting"
  );
  statusBadge.classList.add(`status-${state}`);
  statusLabel.textContent = label;
}

/**
 * Verify the Flask API is reachable and the model is loaded.
 * @returns {Promise<boolean>}
 */
async function checkBackendHealth() {
  setBackendStatus("connecting", "Connecting…");

  try {
    const response = await fetch(API_HEALTH_URL, {
      method: "GET",
      mode: "cors",
      credentials: "omit",
    });

    if (!response.ok) {
      throw new Error(`Health check failed (${response.status})`);
    }

    const data = await response.json();

    backendReady = true;
    modelLoaded = Boolean(data.model_loaded);

    if (!modelLoaded) {
      setBackendStatus("degraded", "Model unavailable");
      showError(
        "The analysis model is not loaded. Run python train_model.py in the backend folder, then restart Flask."
      );
      return false;
    }

    setBackendStatus("online", "Online");
    hideError();
    return true;
  } catch {
    backendReady = false;
    modelLoaded = false;
    setBackendStatus("offline", "Backend offline");
    return false;
  }
}

/**
 * Format current time for message timestamps.
 * @returns {string}
 */
function getTimeString() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/**
 * Scroll chat area to the latest message.
 */
function scrollToBottom() {
  requestAnimationFrame(() => {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  });
}

/**
 * Show an error alert above the input.
 * @param {string} message
 */
function showError(message) {
  errorMessage.textContent = message;
  errorAlert.classList.remove("d-none");
  errorAlert.classList.add("show");
}

/**
 * Hide the error alert.
 */
function hideError() {
  errorAlert.classList.add("d-none");
  errorAlert.classList.remove("show");
}

/**
 * Set loading state on input and send button.
 * @param {boolean} loading
 */
function setLoading(loading) {
  isProcessing = loading;
  sendBtn.disabled = loading;
  symptomInput.disabled = loading;
  VoiceInput.setEnabled(!loading);

  if (loading) {
    VoiceInput.stop();
    sendBtnText.classList.add("d-none");
    sendBtnSpinner.classList.remove("d-none");
  } else {
    sendBtnText.classList.remove("d-none");
    sendBtnSpinner.classList.add("d-none");
    symptomInput.focus();
  }
}

/**
 * Get severity CSS class based on severity string.
 * @param {string} severity
 * @returns {string}
 */
function getSeverityClass(severity) {
  const level = (severity || "").toLowerCase();
  if (level.includes("low") || level.includes("mild")) return "severity-low";
  if (level.includes("moderate") || level.includes("medium")) return "severity-moderate";
  if (level.includes("high") || level.includes("severe") || level.includes("critical")) {
    return "severity-high";
  }
  return "severity-unknown";
}

/**
 * Create a message row element.
 * @param {"user"|"bot"} role
 * @param {string} htmlContent - Inner HTML for the bubble
 * @returns {HTMLElement}
 */
function createMessageElement(role, htmlContent) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.innerHTML =
    role === "bot"
      ? '<i class="bi bi-robot"></i>'
      : '<i class="bi bi-person-fill"></i>';

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.innerHTML = `${htmlContent}<span class="message-time">${getTimeString()}</span>`;

  row.appendChild(avatar);
  row.appendChild(bubble);

  return row;
}

/**
 * Add a user message bubble.
 * @param {string} text
 */
function addUserMessage(text) {
  const row = createMessageElement("user", escapeHtml(text));
  chatMessages.appendChild(row);
  scrollToBottom();
}

/**
 * Add a bot message bubble.
 * @param {string} html
 * @param {{ delay?: number }} options
 * @returns {Promise<void>}
 */
function addBotMessage(html, options = {}) {
  const { delay = 0 } = options;

  return new Promise((resolve) => {
    setTimeout(() => {
      const row = createMessageElement("bot", html);
      chatMessages.appendChild(row);
      scrollToBottom();
      resolve();
    }, delay);
  });
}

/**
 * Show typing indicator and return element for removal.
 * @returns {HTMLElement}
 */
function showTypingIndicator() {
  const row = document.createElement("div");
  row.className = "message-row bot typing-row";
  row.innerHTML = `
    <div class="message-avatar"><i class="bi bi-robot"></i></div>
    <div class="message-bubble">
      <div class="typing-indicator" aria-label="Assistant is typing">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  chatMessages.appendChild(row);
  scrollToBottom();
  return row;
}

/**
 * Remove typing indicator element.
 * @param {HTMLElement} typingRow
 */
function removeTypingIndicator(typingRow) {
  if (typingRow && typingRow.parentNode) {
    typingRow.parentNode.removeChild(typingRow);
  }
}

/**
 * Escape HTML to prevent XSS in user messages.
 * @param {string} text
 * @returns {string}
 */
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Build HTML for prediction result inside a bot bubble.
 * @param {object} data - { predicted_disease, specialist, severity }
 * @param {string[]} symptoms
 * @returns {string}
 */
function buildResultHtml(data, symptoms) {
  const { predicted_disease, specialist, severity } = data;
  const severityClass = getSeverityClass(severity);
  const symptomList = symptoms.map((s) => escapeHtml(s)).join(", ");

  return `
    <p>Based on your symptoms (<strong>${symptomList}</strong>), here is my analysis:</p>
    <div class="result-card">
      <div class="result-item">
        <div class="result-icon disease"><i class="bi bi-virus"></i></div>
        <div>
          <div class="result-label">Possible Condition</div>
          <div class="result-value">${escapeHtml(predicted_disease)}</div>
        </div>
      </div>
      <div class="result-item">
        <div class="result-icon specialist"><i class="bi bi-person-badge"></i></div>
        <div>
          <div class="result-label">Recommended Specialist</div>
          <div class="result-value">${escapeHtml(specialist)}</div>
        </div>
      </div>
      <div class="result-item">
        <div class="result-icon severity"><i class="bi bi-speedometer2"></i></div>
        <div>
          <div class="result-label">Severity Level</div>
          <div class="result-value">
            <span class="severity-badge ${severityClass}">${escapeHtml(severity)}</span>
          </div>
        </div>
      </div>
    </div>
    <p class="mb-0 mt-2"><small>Want to describe more symptoms? Start a new message below — I'll remember everything in this session until then.</small></p>
  `;
}

/**
 * Validate backend response shape.
 * @param {object} data
 * @returns {boolean}
 */
function isValidResponse(data) {
  return (
    data &&
    typeof data === "object" &&
    typeof data.predicted_disease === "string" &&
    typeof data.specialist === "string" &&
    typeof data.severity === "string" &&
    data.predicted_disease.trim() !== "" &&
    data.specialist.trim() !== "" &&
    data.severity.trim() !== ""
  );
}

/**
 * Send symptoms to the Flask backend and display results.
 * @param {string[]} symptoms
 * @param {{ showTyping?: boolean, manageLoading?: boolean }} options
 */
async function predictSymptoms(symptoms, options = {}) {
  const { showTyping = true, manageLoading = true } = options;

  hideError();

  const payloadSymptoms = normalizeSymptomsForApi(symptoms);
  if (payloadSymptoms.length === 0) {
    showError("No valid symptoms to analyze. Please describe your symptoms again.");
    return;
  }

  if (manageLoading) {
    setLoading(true);
  }

  const typingRow = showTyping ? showTypingIndicator() : null;

  try {
    if (!backendReady || !modelLoaded) {
      const healthy = await checkBackendHealth();
      if (!healthy) {
        if (backendReady && !modelLoaded) {
          throw new Error(
            "The prediction model is not available. Run python train_model.py in the backend folder, then restart Flask."
          );
        }
        throw new Error(
          "Unable to connect to the analysis server. Start the Flask backend at http://127.0.0.1:5000 and open this page via Live Server (not as a local file)."
        );
      }
    }

    const response = await fetch(API_PREDICT_URL, {
      method: "POST",
      mode: "cors",
      credentials: "omit",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({ symptoms: payloadSymptoms }),
    });

    const responseText = await response.text();
    let data;

    try {
      data = responseText ? JSON.parse(responseText) : null;
    } catch {
      throw new Error("The server returned a response that could not be parsed as JSON.");
    }

    if (!response.ok) {
      const detail =
        (data && (data.error || data.message)) ||
        `Server returned status ${response.status}.`;
      throw new Error(detail);
    }

    if (!isValidResponse(data)) {
      throw new Error(
        "Received an invalid response from the server. Expected predicted_disease, specialist, and severity fields."
      );
    }

    await addBotMessage(buildResultHtml(data, payloadSymptoms));
    ConversationManager.reset();
    updateInputPlaceholder();
  } catch (error) {
    if (isNetworkError(error)) {
      backendReady = false;
      setBackendStatus("offline", "Backend offline");
      showError(
        "Unable to connect to the server. Ensure Flask is running at http://127.0.0.1:5000 and open this page with Live Server (http://127.0.0.1:5500), not as a file:// URL."
      );
      await addBotMessage(
        "I'm sorry, I couldn't reach the analysis server. Please start the Flask backend and open this page through Live Server, then try again."
      );
    } else {
      showError(error.message || "An unexpected error occurred.");
      await addBotMessage(
        `I'm sorry, something went wrong while analyzing your symptoms. ${escapeHtml(error.message || "Please try again.")}`
      );
    }
  } finally {
    if (typingRow) {
      removeTypingIndicator(typingRow);
    }
    if (manageLoading) {
      setLoading(false);
    }
    if (ConversationManager.isAnalyzing()) {
      ConversationManager.abortAnalyzing();
      updateInputPlaceholder();
    }
    scrollToBottom();
  }
}

/**
 * Ask the next follow-up question with conversational pacing.
 * @param {string} questionHtml
 */
async function askFollowUpQuestion(questionHtml) {
  await addBotMessageWithTyping(questionHtml, 1000);
  ConversationManager.markQuestionPresented();
  updateInputPlaceholder();
}

/**
 * Finalize session and send combined symptoms to the backend.
 * @param {string[]} symptoms
 */
async function finalizeAndPredict(symptoms) {
  const allSymptoms =
    symptoms && symptoms.length > 0
      ? symptoms
      : ConversationManager.getCollectedSymptoms();
  const summary = allSymptoms.map((s) => escapeHtml(s)).join(", ");

  ConversationManager.beginAnalyzing();
  updateInputPlaceholder();
  setLoading(true);

  try {
    await addBotMessageWithTyping(
      `Thank you — I have your full symptom profile: <strong>${summary}</strong>. Analyzing everything together now…`,
      1000
    );

    await predictSymptoms(allSymptoms, { showTyping: false, manageLoading: false });
  } finally {
    setLoading(false);
  }
}

/**
 * Handle form submission (initial symptoms or follow-up answers).
 * @param {Event} event
 */
async function handleSubmit(event) {
  event.preventDefault();

  if (isProcessing || conversationBusy || ConversationManager.isAnalyzing()) return;

  try {
    await handleSubmitFlow(event);
  } catch (error) {
    console.error("Chat submission error:", error);
    showError(error.message || "Something went wrong. Please try again.");
    setLoading(false);
    conversationBusy = false;
  }
}

/**
 * Core submit handler (symptom intake, follow-ups, and prediction).
 * @param {Event} event
 */
async function handleSubmitFlow() {
  conversationBusy = true;
  VoiceInput.stop();
  hideError();

  const rawInput = symptomInput.value.trim();

  if (!rawInput) {
    const emptyMsg = ConversationManager.isAwaitingFollowUp()
      ? "Please answer the follow-up question above, or type \"no\" if none apply."
      : "Please describe at least one symptom before sending.";
    showError(emptyMsg);
    symptomInput.focus();
    conversationBusy = false;
    return;
  }

  try {
    /* ---- Follow-up response: wait for answer before backend ---- */
    if (ConversationManager.isAwaitingFollowUp()) {
      addUserMessage(rawInput);
      ConversationManager.recordMessage("user", rawInput);
      symptomInput.value = "";
      updateInputPlaceholder();

      const result = ConversationManager.processFollowUpResponse(rawInput);
      const memoryHtml = ConversationManager.getMemorySummaryHtml();

      if (result.needsClarification) {
        await addBotMessageWithTyping(result.question, 800);
        ConversationManager.markQuestionPresented();
        return;
      }

      if (result.acknowledged && result.acknowledged.length > 0) {
        await addBotMessageWithTyping(
          `Got it — I've added <strong>${result.acknowledged.map(escapeHtml).join(", ")}</strong>. ${memoryHtml ? `Your symptoms so far: ${memoryHtml}.` : ""}`,
          800
        );
      } else if (result.wasNegative) {
        await addBotMessageWithTyping(
          `No problem. ${memoryHtml ? `I'll keep tracking: ${memoryHtml}.` : "Let's continue."}`,
          650
        );
      } else if (SymptomNLP.extractSymptoms(rawInput).length === 0 && !result.wasNegative) {
        await addBotMessageWithTyping(
          `Thanks for sharing that. ${memoryHtml ? `So far I have: ${memoryHtml}.` : ""}`,
          600
        );
      }

      if (result.needsFollowUp) {
        await askFollowUpQuestion(result.question);
        return;
      }

      await finalizeAndPredict(ConversationManager.getCollectedSymptoms());
      return;
    }

    /* ---- Initial intake or add to existing session memory ---- */
    const symptoms = parseSymptoms(rawInput);

    if (symptoms.length === 0) {
      showError(
        'I couldn\'t identify any symptoms. Try phrases like "I have cold and cough" or "I also have fever".'
      );
      symptomInput.focus();
      return;
    }

    addUserMessage(rawInput);
    ConversationManager.recordMessage("user", rawInput);
    symptomInput.value = "";

    if (ConversationManager.hasActiveMemory()) {
      const appendResult = ConversationManager.appendSymptoms(symptoms);
      const memoryHtml = ConversationManager.getMemorySummaryHtml();

      if (appendResult.newlyAdded.length > 0) {
        await addBotMessageWithTyping(
          `I've added <strong>${appendResult.newlyAdded.map(escapeHtml).join(", ")}</strong> to your list. Your symptoms so far: ${memoryHtml}.`,
          800
        );
      } else {
        await addBotMessageWithTyping(
          `Those symptoms are already in your list: ${memoryHtml}.`,
          700
        );
      }

      if (appendResult.needsFollowUp && appendResult.question) {
        await askFollowUpQuestion(appendResult.question);
      } else if (!ConversationManager.isAwaitingFollowUp()) {
        await finalizeAndPredict(ConversationManager.getCollectedSymptoms());
      }
      return;
    }

    const session = ConversationManager.beginNewSession(symptoms);
    updateInputPlaceholder();

    const memoryHtml = ConversationManager.getMemorySummaryHtml();
    const questionNote =
      session.totalQuestions > 1
        ? `I have <strong>${session.totalQuestions}</strong> quick follow-up questions — we'll go one at a time.`
        : "I have one quick follow-up question.";

    await addBotMessageWithTyping(
      `Thanks for sharing. I've recorded: ${memoryHtml}. ${questionNote}`,
      1000
    );

    if (session.needsFollowUp && session.question) {
      await askFollowUpQuestion(session.question);
      return;
    }

    await finalizeAndPredict(ConversationManager.getCollectedSymptoms());
  } finally {
    conversationBusy = false;
  }
}

/* Event Listeners */
chatForm.addEventListener("submit", handleSubmit);
dismissError.addEventListener("click", hideError);

symptomInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

/* Initialize on page load */
document.addEventListener("DOMContentLoaded", async () => {
  VoiceInput.init({
    inputEl: symptomInput,
    micBtn,
    micIcon,
    statusEl: voiceStatus,
    statusTextEl: voiceStatusText,
  });

  updateInputPlaceholder();
  initChat();
  await checkBackendHealth();
  symptomInput.focus();
});
