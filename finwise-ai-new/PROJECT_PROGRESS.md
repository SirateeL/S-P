# FinWise AI - Project Progress Summary

## Current status

**FinWise AI is fully accessible on the developer's MacBook, but it is not yet deployed online.** Teammates can access the source code through GitHub, but they cannot yet use the live website when the MacBook is off.

## 1. What we have completed

1. **Built the FinWise AI website**
   The application includes an Overview dashboard, Finance AI Expert, Budget Planner, Savings Goals, and Learning Hub. It uses Python Flask for the backend, Ollama for AI, and ChromaDB for storing searchable PDF passages.

2. **Fixed the PDF question-answering system**
   Initially, FinWise AI sometimes failed to answer questions even when the information existed in uploaded PDFs. We used Codex to improve document retrieval and citation validation. The AI can now answer questions using PDF evidence and display the relevant sources.

3. **Uploaded the project to GitHub**
   The code is in the team's repository: https://github.com/SirateeL/S-P/tree/main/finwise-ai-new

4. **Prepared the application for deployment**
   Codex added initial security improvements, including password protection, restricted PDF access, CSRF protection, and production configuration. Codex reported that 81 automated tests passed. One issue remains: Gunicorn could not be installed in the existing environment, so the production server has not been verified.

5. **Created and tested an Admin login**
   A local admin password was configured through Terminal. Login with the username `admin` works and opens the FinWise AI homepage.

6. **Successfully tested the AI and PDF citations**
   Question: "Explain the DCF valuation process." FinWise AI generated an answer and displayed citations to PDF pages 6 and 16. Opening the cited PDFs also worked.
   One quality issue remains: the AI repeated essentially the same explanation several times.

## 2. What is working now

| Feature                     | Status                                   |
| --------------------------- | ---------------------------------------- |
| Homepage and dashboard      | Working                                  |
| Admin login                 | Working                                  |
| Ollama AI connection        | Working locally                          |
| PDF upload and retrieval    | Working                                  |
| AI answers with citations   | Working                                  |
| Budget and savings features | Displayed; not fully retested after changes |
| GitHub repository           | Uploaded earlier                         |
| Public website URL          | Not yet                                  |
| 24/7 online availability    | Not yet                                  |

## 3. Why teammates cannot open the website yet

FinWise AI currently runs at `http://127.0.0.1:5001`. This is a localhost address, meaning the application runs on the developer's own MacBook (Flask + Ollama + PDFs). If the MacBook is shut down, the website stops. The GitHub code remains available.

## 4. What we still need to do

The goal is to let teammates open FinWise AI through a normal website link, even when the MacBook is turned off.

| Remaining task                      | Status             |
| ----------------------------------- | ------------------ |
| Choose affordable cloud hosting     | Not finalized      |
| Connect Flask to cloud              | Not started        |
| Connect a cloud-accessible AI model | Not started        |
| Set up permanent PDF storage        | Not started        |
| Configure secure online access      | Partially prepared |
| Upload updated code to GitHub       | In progress        |
| Test the public website             | Not started        |

We considered Render for Flask hosting and a cloud AI API instead of local Ollama. Free hosting has limitations, particularly for persistent PDF storage and AI availability. A Render account problem stopped that attempt.

## 5. What should we do next?

**Option A - Finish the university submission.** Use the working local website to record a demonstration video and submit the GitHub repository and required report. This avoids unnecessary deployment work if a live URL is not required.

**Option B - Continue deploying FinWise AI.** Resolve the hosting account problem, select an AI provider and persistent storage, and deploy. This requires more work and may involve costs.

The project itself is working; only the online deployment is unfinished.
