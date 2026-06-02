/* Plain-language helpers for user-facing messages */

function friendlyError(msg) {
  if (!msg) return 'Something went wrong. Please try again.';
  const m = String(msg).toLowerCase();
  if (m.includes('xsmtpsib') || m.includes('xkeysib') || m.includes('brevo')) {
    return 'We could not send email right now. Please ask your administrator to check the email setup.';
  }
  if (m.includes('invalid email or password')) return 'That email or password is not correct.';
  if (m.includes('internal server error') || m.includes('failed to fetch') || m.includes('network')) {
    return 'Could not reach the server. Make sure python api.py is running, then hard refresh this page.';
  }
  if (m.includes('already exists')) return 'An account with this email already exists. Try signing in instead.';
  if (m.includes('verification code') || m.includes('invalid or expired')) {
    return 'That code is wrong or has expired. Tap Resend code to get a new one.';
  }
  if (m.includes('not authenticated') || m.includes('expired token')) {
    return 'Your session ended. Please sign in again.';
  }
  return msg.replace(/Brevo|API|OTP|SMTP|backend|FastAPI|CSV|SKU|ensemble|ARIMA|Prophet|MLP|\.env/gi, '').trim()
    || 'Something went wrong. Please try again.';
}

function friendlyAlertMessage(result) {
  if (!result?.message) return '';
  if (result.status === 'sent') return `We emailed you about ${result.sent_count || ''} low-stock item(s).`;
  if (result.status === 'no_alerts') return 'Everything looks good — nothing needs attention right now.';
  if (result.status === 'cooldown') return 'We already emailed you about these items recently.';
  return result.message;
}
