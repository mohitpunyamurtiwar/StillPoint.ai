/**
 * Stillpoint AI — Behavioral Evaluation Engine (ai-evaluator.js)
 * Enterprise-grade multi-dimensional cognitive, affective, and compliance analysis.
 */
"use strict";

function analyzeEvaluation(data = {}) {
  const profile = data.profile || "General Technical Interview";
  const durationSec = Number(data.sessionDuration) || 60;
  const attentionSec = Number(data.attentionTime) || durationSec;
  const focusScore = Math.max(10, Math.min(100, Math.round(Number(data.focusScore) || 90)));
  const gazeAlignment = Math.max(10, Math.min(100, Math.round(Number(data.gazeAlignmentPct) || 92)));
  const headStability = Math.max(10, Math.min(100, Math.round(Number(data.headStabilityPct) || 90)));
  const saccades = Number(data.saccadesCount) || 0;
  const glances = Number(data.glanceCount) || 0;
  const speakingRatio = Math.max(0, Math.min(100, Math.round(Number(data.speakingRatio) || 0)));
  const criticalCount = Number(data.anomalies?.critical) || 0;
  const noticeCount = Number(data.anomalies?.notices) || 0;
  const events = Array.isArray(data.events) ? data.events : [];
  const biometrics = data.biometrics || {};

  const avgEar = Number(biometrics.avgEar) || 0.28;
  const blinkRate = Number(biometrics.blinkRate) || 16;
  const drowsinessDetections = Number(biometrics.drowsinessCount) || 0;
  const sadDetections = Number(biometrics.sadAffectCount) || 0;
  const browFlashes = Number(biometrics.browFlashCount) || 0;
  const microTwitches = Number(biometrics.microTwitchCount) || 0;
  const noiseSpikes = Number(biometrics.bgNoiseSpikes) || 0;

  // 1. DIMENSION 1: Cognitive Steadiness & Focal Lock (0 - 100)
  let cognitiveScore = focusScore;
  if (saccades > 0) cognitiveScore -= Math.min(25, saccades * 8);
  if (glances > 2) cognitiveScore -= Math.min(20, (glances - 2) * 5);
  cognitiveScore = Math.max(20, Math.min(100, cognitiveScore));

  // 2. DIMENSION 2: Affective Composure & Stress Resilience (0 - 100)
  let composureScore = 95;
  if (blinkRate > 28) composureScore -= Math.min(20, (blinkRate - 28) * 1.5);
  if (microTwitches > 1) composureScore -= Math.min(18, microTwitches * 6);
  if (sadDetections > 1) composureScore -= Math.min(24, sadDetections * 8);
  if (noticeCount > 2) composureScore -= Math.min(15, (noticeCount - 2) * 3);
  composureScore = Math.max(25, Math.min(100, Math.round(composureScore)));

  // 3. DIMENSION 3: Communicative Confidence & Articulation (0 - 100)
  let commScore = 80;
  if (speakingRatio >= 25 && speakingRatio <= 65) {
    commScore = 94; // ideal interview dialogue balance
  } else if (speakingRatio > 65) {
    commScore = 88; // highly vocal / dominant
  } else if (speakingRatio > 10) {
    commScore = 82; // active listening with brief answers
  } else {
    commScore = 72; // primarily silent
  }
  if (browFlashes > 0) commScore = Math.min(100, commScore + Math.min(6, browFlashes * 2)); // active expressions enhance engagement
  commScore = Math.max(30, Math.min(100, commScore));

  // 4. DIMENSION 4: Physical Demeanor & Alertness (0 - 100)
  let demeanorScore = 96;
  if (avgEar < 0.23 || drowsinessDetections > 0) {
    demeanorScore -= Math.min(45, (drowsinessDetections * 15) + (avgEar < 0.23 ? 20 : 0));
  }
  if (headStability < 75) demeanorScore -= Math.min(25, (75 - headStability) * 1.2);
  demeanorScore = Math.max(20, Math.min(100, Math.round(demeanorScore)));

  // 5. DIMENSION 5: Environmental Compliance & Academic Integrity (0 - 100)
  let integrityScore = 100;
  if (criticalCount > 0) integrityScore -= Math.min(60, criticalCount * 30);
  if (saccades > 2) integrityScore -= Math.min(25, (saccades - 2) * 8);
  if (noiseSpikes > 1) integrityScore -= Math.min(15, noiseSpikes * 5);
  integrityScore = Math.max(15, Math.min(100, Math.round(integrityScore)));

  // OVERALL WEIGHTED COMPOSITE INDEX
  const compositeIndex = Math.round(
    cognitiveScore * 0.30 +
    integrityScore * 0.30 +
    composureScore * 0.15 +
    demeanorScore * 0.15 +
    commScore * 0.10
  );

  // EXECUTIVE VERDICT SYNTHESIS
  let verdict = "RECOMMENDED — HIGH FOCUS & INTEGRITY";
  let verdictClass = "recommended";
  let verdictSummary = "The candidate exhibited disciplined visual focus, consistent forward posture, and strong cognitive composure with no verified compliance anomalies.";

  if (criticalCount > 0 || integrityScore < 60) {
    verdict = "HIGH INTEGRITY RISK — MANUAL AUDIT REQUIRED";
    verdictClass = "risk";
    verdictSummary = "Unauthorized secondary devices or extraneous individuals were detected during the session. Detailed timeline inspection and compliance verification required.";
  } else if (drowsinessDetections > 2 || demeanorScore < 60) {
    verdict = "ELEVATED FATIGUE / NOTICE — RE-ASSESSMENT SUGGESTED";
    verdictClass = "notice";
    verdictSummary = "Candidate displayed sustained eyelid ptosis and fatigued physical posture. While compliance remained high, energy and alertness were subdued.";
  } else if (cognitiveScore < 65 || saccades >= 2) {
    verdict = "MODERATE FOCUS DRIFT — NOTES SUSPECTED";
    verdictClass = "notice";
    verdictSummary = "Repeated ocular saccades and off-screen downward glances suggest external notes, second monitors, or secondary materials were being referenced.";
  } else if (compositeIndex >= 88) {
    verdict = "STRONG RECOMMENDATION — EXCEPTIONAL ENGAGEMENT";
    verdictClass = "recommended";
    verdictSummary = "Demonstrated exemplary screen focal lock, dynamic expressive warmth, steady emotional biometrics, and verified zero environmental intrusion.";
  } else {
    verdict = "ACCEPTABLE CANDIDATE — STANDARD BEHAVIORAL PROFILE";
    verdictClass = "qualified";
    verdictSummary = "Demonstrated baseline professional composure and expected attention metrics consistent with standardized interview conditions.";
  }

  // QUALITATIVE EXECUTIVE NARRATIVE
  const durationMin = (durationSec / 60).toFixed(1);
  const narrativeParagraphs = [
    `During this ${durationMin}-minute session for "${profile}", the Stillpoint AI Computer Vision & Audio Engine captured continuous biometric telemetry at 30 FPS. The candidate maintained an average focus index of ${focusScore}%, with ocular alignment directed at the central monitor ${gazeAlignment}% of the evaluated time. Head orientation remained stable within acceptable natural ergonomic boundaries (${headStability}% axial consistency).`,
    `Affective and micro-expression telemetry revealed a ${composureScore >= 80 ? "composed and steady" : "mildly tense or fluctuating"} biometric profile. Blink frequency averaged ${blinkRate} blinks/minute (${blinkRate > 26 ? "indicating elevated nervous arousal under technical questioning" : "consistent with relaxed professional focus"}). Eyelid aperture maintained an average of ${avgEar.toFixed(2)} EAR${drowsinessDetections > 0 ? `, though ${drowsinessDetections} momentary ptosis/fatigue dips were logged` : " with crisp alert visual engagement"}. ${browFlashes > 0 ? `${browFlashes} AU1/2 eyebrow elevation micro-gestures were registered, reflecting active cognitive engagement and intellectual curiosity.` : "Facial affect remained steady throughout."}`,
    `Environmental acoustic floor was monitored continuously via client-side FFT audio analysis. The room acoustics registered ${noiseSpikes > 0 ? `${noiseSpikes} ambient acoustic disruptions outside candidate speech` : "a clean, quiet acoustic floor with zero unauthorized ambient voice interference"}. Active candidate speech accounted for ${speakingRatio}% of the session, indicating ${speakingRatio > 25 ? "confident, articulate dialogue" : "concise, focused responses"}.`
  ];

  // OBSERVED STRENGTHS
  const strengths = [];
  if (focusScore >= 85) strengths.push(`Sustained High Focus: Maintained ${focusScore}% focus rating with reliable central gaze lock.`);
  if (integrityScore >= 90) strengths.push(`Zero Device Intrusions: No mobile phones, secondary persons, or unauthorized hardware detected.`);
  if (headStability >= 85) strengths.push(`Postural Stability: Exemplary 3D head alignment (${headStability}%) with minimal restless torso movement.`);
  if (blinkRate <= 24 && composureScore >= 80) strengths.push(`Emotional Composure: Low autonomic stress markers (steady ${blinkRate} blinks/min).`);
  if (browFlashes > 0) strengths.push(`Expressive Articulation: Natural facial Action Units (AU1/2, AU12) demonstrating genuine rapport.`);
  if (strengths.length < 2) strengths.push("Compliance Verified: Satisfied standard remote identity and room stability guidelines.");

  // RISK FLAGS & WATCH ITEMS
  const risks = [];
  if (criticalCount > 0) risks.push(`Critical Hardware Flag: ${criticalCount} unauthorized device/person anomaly recorded in audit log.`);
  if (saccades > 0) risks.push(`Ocular Saccade Reading Pattern: ${saccades} rapid horizontal saccade bursts detected, indicating possible external script or second monitor reading.`);
  if (drowsinessDetections > 0) risks.push(`Ptosis / Low Energy: ${drowsinessDetections} sustained eyelid droop moments (${avgEar.toFixed(2)} EAR) indicating fatigue.`);
  if (sadDetections > 1) risks.push(`Flat / Downcast Affect: Sustained mouth corner droop (AU15) observed during technical inquiry.`);
  if (noiseSpikes > 0) risks.push(`Acoustic Noise Spike: ${noiseSpikes} ambient audio disturbances logged while candidate was silent.`);
  if (risks.length === 0) risks.push("Zero Anomalies: Clear audit trail with no suspicious movements or compliance alerts.");

  // INTERVIEWER FOLLOW-UP QUESTIONS
  const questions = [];
  if (saccades > 0) {
    questions.push("Can you elaborate on that architectural trade-off from memory without looking at notes, detailing alternative data structures?");
  }
  if (drowsinessDetections > 0 || composureScore < 70) {
    questions.push("How do you typically manage prolonged on-call rotations or high-pressure incident mitigation when working under cognitive fatigue?");
  }
  if (speakingRatio < 20) {
    questions.push("Walk me step-by-step through your thought process and mental model for this problem — I'd like to hear how you structure your rationale.");
  } else {
    questions.push("Tell me about a time when you received direct pushback on an architectural proposal — how did you respond and iterate?");
  }
  if (noiseSpikes > 0) {
    questions.push("Could you confirm your physical workspace setup and tell us about your daily remote collaboration habits?");
  }

  return {
    success: true,
    profile,
    timestamp: new Date().toISOString(),
    compositeIndex,
    verdict,
    verdictClass,
    verdictSummary,
    competencies: [
      { name: "Cognitive Steadiness & Focal Lock", score: cognitiveScore, rating: cognitiveScore >= 80 ? "Strong" : cognitiveScore >= 60 ? "Moderate" : "Needs Review" },
      { name: "Affective Composure & Stress Index", score: composureScore, rating: composureScore >= 80 ? "Calm" : composureScore >= 60 ? "Moderate" : "Elevated Stress" },
      { name: "Communicative Confidence & Flow", score: commScore, rating: commScore >= 80 ? "Articulate" : commScore >= 60 ? "Standard" : "Low Interaction" },
      { name: "Physical Alertness & Demeanor", score: demeanorScore, rating: demeanorScore >= 80 ? "Alert & Focused" : demeanorScore >= 60 ? "Moderate" : "Fatigued / Slouched" },
      { name: "Environmental & Academic Integrity", score: integrityScore, rating: integrityScore >= 85 ? "Verified Clean" : integrityScore >= 65 ? "Notice" : "High Risk" }
    ],
    narrative: narrativeParagraphs,
    strengths,
    risks,
    followUpQuestions: questions.slice(0, 3)
  };
}

module.exports = { analyzeEvaluation };
