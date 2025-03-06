function formatDateSeparator(messageDate) {
    const today = new Date(); // Current date in local time
  
    const messageYear = messageDate.getFullYear();
    const messageMonth = messageDate.getMonth(); // 0-based
    const messageDay = messageDate.getDate();
  
    const todayYear = today.getFullYear();
    const todayMonth = today.getMonth();
    const todayDay = today.getDate();
  
    // Check if the message is from today
    if (
      messageYear === todayYear &&
      messageMonth === todayMonth &&
      messageDay === todayDay
    ) {
      return 'Today';
    }
  
    // Check if the message is from yesterday
    const yesterday = new Date(today);
    yesterday.setDate(today.getDate() - 1);
    if (
      messageYear === yesterday.getFullYear() &&
      messageMonth === yesterday.getMonth() &&
      messageDay === yesterday.getDate()
    ) {
      return 'Yesterday';
    }
  
    // Array of month names for formatting
    const months = [
      'January', 'February', 'March', 'April', 'May', 'June',
      'July', 'August', 'September', 'October', 'November', 'December'
    ];
  
    // Same year: "Day Month"
    if (messageYear === todayYear) {
      return `${messageDay} ${months[messageMonth]}`;
    }
  
    // Different year: "Day Month, Year"
    return `${messageDay} ${months[messageMonth]}, ${messageYear}`;
  }

  function formatDuration(seconds) {
    if (seconds === 0) return "0s";
    const secondsN = Math.round(seconds)
    const hours = Math.floor(secondsN / 3600);
    const minutes = Math.floor((secondsN % 3600) / 60);
    const secs = secondsN % 60;
  
    let result = "";
    if (hours > 0) {
      result += `${hours}h`;
    }
    if (minutes > 0) {
      if (result) result += " "; // Add space if there's already a unit
      result += `${minutes}m`;
    }
    if (secs > 0 || (!hours && !minutes)) {
      if (result) result += " "; // Add space if there's already a unit
      result += `${secs}s`;
    }
  
    return result;
  }

  export { formatDateSeparator, formatDuration }
