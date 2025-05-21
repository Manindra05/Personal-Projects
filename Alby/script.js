
// Track if the user picked a topic yet
let topicChosen = false;

// Track if the user picked a mode yet (general or roleplay)
let modeChosen = false;

// The current topic we are talking about
let currentTopic = "";

// The current mode ("general" or "roleplay")
let mode = "";

// Store the user's last answer
let lastUserAnswer = "";

// Speech Recognition variables
let recognition;
let finalTranscript = "";



const apiKey = "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";

// Setting up speech recognition
window.SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

// Start listening when user presses Mic button
function startListening() {
  recognition = new SpeechRecognition();
  recognition.lang = "en-US"; // Recognize English
  recognition.interimResults = true; // Show partial results while speaking
  recognition.continuous = false; // Stop after each speech

  finalTranscript = "";

  document.getElementById("albyImage").classList.add("bounce", "glow");

  recognition.start();

  recognition.onresult = function(event) {
    let interimTranscript = "";
    for (let i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        finalTranscript += event.results[i][0].transcript + " ";
      } else {
        interimTranscript += event.results[i][0].transcript;
      }
    }
    document.getElementById("textInput").value = finalTranscript + interimTranscript;
  };

  recognition.onend = function() {
    document.getElementById("albyImage").classList.remove("bounce", "glow");

    const spokenText = document.getElementById("textInput").value.trim();
    if (spokenText !== "") {
      handleUserInput(spokenText);
      document.getElementById("textInput").value = "";
    }
  };

  recognition.onerror = function(event) {
    console.error("Speech recognition error:", event.error);
  };
}

// --- TEXT INPUT HANDLING ---

// Handle when user manually types and presses Send
function sendText() {
  const textInput = document.getElementById("textInput").value.trim();
  if (textInput === "") return;
  handleUserInput(textInput);
  document.getElementById("textInput").value = "";
}

// --- TOPIC + MODE SELECTION ---

// Called when user clicks a Topic button
function selectTopic(selectedTopic) {
  currentTopic = selectedTopic;
  topicChosen = true;
  document.getElementById("topicSelection").style.display = "none";
  document.getElementById("modeSelection").style.display = "block";
}

// Called when user picks General or Roleplay Mode
function selectMode(selectedMode) {
  mode = selectedMode;
  modeChosen = true;
  document.getElementById("modeSelection").style.display = "none";
  generateFirstMessage(currentTopic, mode);
}


// --- USER INPUT AND CONVERSATION FLOW ---

// Handle what happens when user inputs something
function handleUserInput(userInput) {
  lastUserAnswer = userInput;

  if (!topicChosen) {
    // If user typed a custom topic (without clicking buttons)
    topicChosen = true;
    modeChosen = true;
    currentTopic = userInput;
    mode = "general";
    document.getElementById("topicSelection").style.display = "none";
    document.getElementById("modeSelection").style.display = "none";
    generateFirstMessage(currentTopic, mode);
  } else {
    // Otherwise continue normal conversation
    showUserAnswer(userInput);
    correctUserAnswer(userInput);
  }
}

// Define roleplay behavior based on topic
function getRoleplayIntro(topic) {
  if (topic === "Doctor Visits") {
    return "You are roleplaying as a doctor. Greet the patient and ask what symptoms they have.";
  } else if (topic === "Job Interviews") {
    return "You are roleplaying as a job interviewer. Start by asking the candidate to tell you about themselves.";
  } else if (topic === "Ordering at a Restaurant") {
    return "You are roleplaying as a restaurant waiter. Greet the customer and ask for their order.";
  } else if (topic === "Making Friends" || topic === "Daily Conversation") {
    return "You are roleplaying as a new friend meeting someone casually for the first time.";
  } else {
    return "You are a friendly conversation partner.";
  }
}


// Generate the first message Alby sends based on topic and mode
async function generateFirstMessage(topic, mode) {
  let systemContent = "";
  if (mode === "roleplay") {
    systemContent = getRoleplayIntro(topic);
  } else {
    systemContent = "You are a friendly English tutor. Start a casual conversation based on the topic.";
  }

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model: "gpt-4-1106-preview",
      messages: [
        { role: "system", content: systemContent },
        { role: "user", content: `Start a conversation about: ${topic}` }
      ],
      temperature: 0.7
    })
  });

  const data = await response.json();
  const question = data.choices[0].message.content;

  document.getElementById("responseContainer").innerHTML = `<p><strong>Alby says:</strong> ${question}</p>`;
}

// shows what the user said 
function showUserAnswer(text) {
  document.getElementById("responseContainer").innerHTML += `
    <p><strong>You said:</strong> "${text}"</p>
    <p><em>Correcting...</em></p>
  `;
}


// corrects their english if needed and encourage them 
async function correctUserAnswer(userText) {
  try {
    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: "gpt-4-1106-preview",
        messages: [
          { role: "system", content: "You are a friendly English tutor. Correct the user's English naturally, encourage them, and ask a related follow-up question to continue the conversation." },
          { role: "user", content: `Correct this English: "${userText}" and then encourage me and ask a related follow-up question.` }
        ],
        temperature: 0.6
      })
    });

    const data = await response.json();

    if (data.error) {
      document.getElementById("responseContainer").innerHTML += `<p><strong>Error:</strong> ${data.error.message}</p>`;
      return;
    }

    const correction = data.choices[0].message.content;

    document.getElementById("responseContainer").innerHTML += `
      <p><strong>Alby's feedback:</strong> ${correction}</p>
    `;

    document.getElementById("actionButtons").style.display = "flex";

  } catch (error) {
    console.error(error);
    document.getElementById("responseContainer").innerHTML += `<p><strong>Error:</strong> Could not connect to OpenAI.</p>`;
  }
}

function moveOn() {
  generateFirstMessage(currentTopic, mode);
  document.getElementById("actionButtons").style.display = "none";
}

function tryAgain() {
  document.getElementById("responseContainer").innerHTML += `
    <p>🔄 Alby says: Try again, you're doing great!</p>
  `;
  document.getElementById("actionButtons").style.display = "none";
}

// reset everything to default for a new topic 

function resetTopic() {
  topicChosen = false;
  modeChosen = false;
  currentTopic = "";
  mode = "";
  lastUserAnswer = "";
  document.getElementById("responseContainer").innerHTML = "";
  document.getElementById("actionButtons").style.display = "none";
  document.getElementById("topicSelection").style.display = "block";
  document.getElementById("modeSelection").style.display = "none";
}
