      // --- Configuration ---
      const apiUrl = "https://irtis-aw.fi.muni.cz/model_response/";
      const personalityMap = {
        I: "Introversion (I) – You likely draw energy from your inner world of thoughts and ideas.",
        N: "Intuition (N) – You tend to focus on possibilities, patterns, and future implications.",
        T: "Thinking (T) – Decisions are often based on logic and objective analysis.",
        J: "Judging (J) – You probably prefer structure, plans, and reaching closure.",
        E: "Extraversion (E) – You likely gain energy from interacting with people and the outer world.",
        S: "Sensing (S) – You tend to focus on concrete facts, details, and present reality.",
        F: "Feeling (F) – Decisions are often guided by values and consideration for others.",
        P: "Perceiving (P) – You probably prefer flexibility, spontaneity, and keeping options open.",
      };
      const MAX_CHARS = 500; // Define the limit

      // --- DOM Element References ---
      const form = document.getElementById("personalityForm");
      const queryTextarea = document.getElementById("queryText");
      const submitButton = document.getElementById("submitButton");
      const loadingIndicator = document.getElementById("loadingIndicator");
      const errorMessageDiv = document.getElementById("errorMessage");
      const resultAreaDiv = document.getElementById("resultArea");
      const resultCodeH2 = document.getElementById("resultCode");
      const resultExplanationList = document.getElementById(
        "resultExplanationList",
      );
      const charCounter = document.getElementById("charCounter");

      // --- State Helper ---
      let isLoading = false;

      // --- Event Listeners ---
      form.addEventListener("submit", handleSubmit);
      queryTextarea.addEventListener("input", handleTextInput);

      // --- Functions ---

      function handleTextInput() {
        const currentLength = queryTextarea.value.length;
        charCounter.textContent = `${currentLength} / ${MAX_CHARS}`;

        if (currentLength > MAX_CHARS) {
          charCounter.classList.add("limit-exceeded");
          submitButton.disabled = true; // Disable button if over limit
        } else {
          charCounter.classList.remove("limit-exceeded");
          // Only enable button if not loading AND within limit
          submitButton.disabled = isLoading;
        }
      }

      async function handleSubmit(event) {
        event.preventDefault();
        if (isLoading) return;

        const query = queryTextarea.value.trim();
        const currentLength = queryTextarea.value.length; // Use untrimmed length for check

        // Basic validation
        if (!query) {
          showError("Please enter some text to analyze.");
          return;
        }

        // Check character limit before submitting
        if (currentLength > MAX_CHARS) {
          showError(`Input cannot exceed ${MAX_CHARS} characters.`);
          handleTextInput(); // Ensure counter reflects error state
          return;
        }

        setLoadingState(true);
        clearResultsAndErrors();

        try {
          const response = await fetch(apiUrl, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Accept: "application/json",
            },
            body: JSON.stringify({ query: query }), // Send trimmed query
          });

          if (!response.ok) {
            let errorBody = null;
            try {
              errorBody = await response.json();
            } catch (e) {
              /* ignore */
            }
            console.error("API Error Response:", errorBody);
            throw new Error(
              `API Error: ${response.status} ${response.statusText}`,
            );
          }

          const data = await response.json();
          const rawPersonalityType = data.response;

          if (typeof rawPersonalityType !== "string") {
            console.error(
              "Unexpected data format in response key:",
              rawPersonalityType,
            );
            throw new Error(
              "Received unexpected data format from the server.",
            );
          }

          const personalityType = rawPersonalityType.trim();

          if (personalityType && /^[A-Z]{4}$/i.test(personalityType)) {
            displayResults(personalityType.toUpperCase());
          } else {
            console.error(
              "Invalid or empty personality type received:",
              personalityType,
            );
            throw new Error(
              "Received an invalid personality type format from the server.",
            );
          }
        } catch (err) {
          console.error("Fetch or Processing error:", err);
          showError(
            err.message || "Failed to get personality type. Please try again.",
          );
        } finally {
          setLoadingState(false);
        }
      }

      function setLoadingState(loading) {
        isLoading = loading;
        const currentLength = queryTextarea.value.length;
        // Disable button if loading OR if over character limit
        submitButton.disabled = loading || currentLength > MAX_CHARS;
        queryTextarea.disabled = loading;
        loadingIndicator.classList.toggle("visible", loading);
        if (loading) {
          submitButton.classList.add("is-loading");
        } else {
          submitButton.classList.remove("is-loading");
        }
      }

      function clearResultsAndErrors() {
        errorMessageDiv.classList.remove("visible");
        resultAreaDiv.classList.remove("visible");
        setTimeout(() => {
          if (!errorMessageDiv.classList.contains("visible")) {
            errorMessageDiv.textContent = "";
          }
          if (!resultAreaDiv.classList.contains("visible")) {
            resultCodeH2.textContent = "";
            resultExplanationList.innerHTML = "";
          }
        }, 500); // Match transition duration (CSS opacity is 0.5s)
      }

      function showError(message) {
        clearResultsAndErrors(); // Clear results first
        errorMessageDiv.textContent = message;
        setTimeout(() => {
          errorMessageDiv.classList.add("visible");
        }, 50); // Short delay
      }

      function displayResults(resultCode) {
        clearResultsAndErrors(); // Clear errors first
        resultCodeH2.textContent = resultCode;
        resultExplanationList.innerHTML = "";

        resultCode.split("").forEach((letter) => {
          const explanationText =
            personalityMap[letter] || "Explanation not found.";
          const listItem = document.createElement("li");
          const strong = document.createElement("strong");
          strong.textContent = `${letter}:`;
          const span = document.createElement("span");
          span.textContent = ` ${explanationText}`;
          listItem.appendChild(strong);
          listItem.appendChild(span);
          resultExplanationList.appendChild(listItem);
        });

        setTimeout(() => {
          resultAreaDiv.classList.add("visible");
        }, 50); // Short delay
      }

      // --- Initial Setup ---
      handleTextInput(); // Call once on load to set initial counter state