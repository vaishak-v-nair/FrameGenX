export function runVerdict(artifacts) {
  const decisions = (artifacts?.attempts || [])
    .map((attempt) => attempt.decision)
    .filter((decision) => decision != null);
  if (decisions.some((decision) => decision.accept === true)) return true;
  if (decisions.length > 0) return false;
  return null;
}

export function apiErrorMessage(body, fallback = 'Request failed.') {
  const detail = body?.detail;
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (typeof detail?.message === 'string' && detail.message.trim()) return detail.message;
  if (Array.isArray(detail)) {
    const messages = detail.slice(0, 3).map(item => item?.msg).filter(msg => typeof msg === 'string' && msg.trim());
    if (messages.length) return messages.join('; ');
  }
  return fallback;
}

export function isVqaUnavailable(verification) {
  if (verification?.vqa_status === 'unavailable' || verification?.vqa_status === 'partial') return true;
  return verification?.vqa_status == null &&
    verification?.vqa_details?.length > 0 &&
    verification.vqa_details.every((item) => item.answer === 'error');
}

export function bestVisibleAttempt(artifacts) {
  const attempts = (artifacts?.attempts || []).filter((attempt) => attempt.image_url);
  const accepted = [...attempts].reverse().find((attempt) => attempt.decision?.accept === true);
  if (accepted) return accepted;
  return [...attempts].sort((a, b) => {
    const failuresA = a.verification?.hard_failures?.length ?? Number.MAX_SAFE_INTEGER;
    const failuresB = b.verification?.hard_failures?.length ?? Number.MAX_SAFE_INTEGER;
    if (failuresA !== failuresB) return failuresA - failuresB;
    return (b.decision?.reward ?? -1) - (a.decision?.reward ?? -1);
  })[0] || null;
}
