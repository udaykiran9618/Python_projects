import speech_recognition as sr
from transformers import pipeline
from datetime import datetime

# Initialize recognizer and summarizer
recognizer = sr.Recognizer()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def listen_and_transcribe():
    with sr.Microphone() as source:
        print("🎤 Speak now... (say 'stop listening' to finish)")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio_data = []
        
        while True:
            try:
                print("🎧 Listening...")
                audio = recognizer.listen(source)
                text = recognizer.recognize_google(audio)
                print(f"You said: {text}")

                if "stop listening" in text.lower():
                    print("🛑 Stopping transcription.")
                    break
                audio_data.append(text)
            except sr.UnknownValueError:
                print("❓ Could not understand audio.")
            except sr.RequestError as e:
                print(f"⚠️ API error: {e}")
                break

        full_text = " ".join(audio_data)
        return full_text

def summarize_text(text):
    if len(text) < 50:
        return "Text too short to summarize."
    print("📝 Summarizing...")
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    return summary

def save_notes(transcript, summary):
    filename = f"auto_notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("🧠 Auto-Generated Notes\n\n")
        f.write("📌 Summary:\n")
        f.write(summary + "\n\n")
        f.write("🗒️ Full Transcript:\n")
        f.write(transcript)
    print(f"✅ Notes saved as {filename}")

if __name__ == "__main__":
    transcript = listen_and_transcribe()

    if not transcript.strip():
        print("❌ No valid speech detected. Exiting.")
    else:
        print("\n🗒️ Full Transcript:\n", transcript)
        summary = summarize_text(transcript)
        print("\n📌 Summary:\n", summary)
        save_notes(transcript, summary)
