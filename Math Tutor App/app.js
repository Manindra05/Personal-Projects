function stripLatex(text) {
  return text.replace(/\\\(|\\\)/g, '');
}

async function askGPTFromHistory(history) {
  try {
    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer {OPENAI API KEY}` // Replace with your actual key
      },
      body: JSON.stringify({
        model: "gpt-3.5-turbo",
        messages: history
      })
    });

    if (!response.ok) {
      const err = await response.text();
      console.error("OpenAI API Error:", err);
      document.getElementById("question").innerText = `Error: ${response.status} - ${err}`;
      return "There was a problem getting the question.";
    }

    const data = await response.json();
    const answer = data.choices[0].message.content;
    return answer;
  } catch (error) {
    console.error("Fetch failed:", error);
    document.getElementById("question").innerText = "Error connecting to GPT.";
    return "Error occurred.";
  }
}