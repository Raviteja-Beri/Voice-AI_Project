import speech_recognition as sr
import pyttsx3 as pt
import pywhatkit as pk
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

load_dotenv()

listening = sr.Recognizer()
engine = pt.init()

# Initializing AI agent
personal_agent = Agent(
    name="Voice-AI",
    model=OpenAIChat(model="gpt-4o-mini"),
    tools=[DuckDuckGo()],
    show_tool_calls=True,
    markdown=True,
    debug_mode=False,
)

def speak(text):
    print(f"Assistant: {text}")
    engine.say(text)
    engine.runAndWait()

def hear():
    cmd = ""
    try:
        with sr.Microphone() as source:
            print("Listening...")
            listening.adjust_for_ambient_noise(source, duration=1)
            audio = listening.listen(source, timeout=5, phrase_time_limit=10)
            cmd = listening.recognize_google(audio)
            cmd = cmd.lower().strip()
            print(f"You said: {cmd}")
    except sr.WaitTimeoutError:
        print("No speech detected")
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand what you said")
    except sr.RequestError as e:
        print(f"Error with speech recognition service: {e}")
    except Exception as e:
        print(f"Error: {e}")
    
    return cmd

def handle_ai_query(cmd):
    try:
        speak("Let me think about that...")
        response = personal_agent.run(cmd)
        if hasattr(response, 'content'):
            ai_response = response.content
        else:
            ai_response = str(response)
        ai_response = ai_response.replace('*', '').replace('#', '')
        
        speak(ai_response)
        return True
    except Exception as e:
        speak("Sorry, I encountered an error processing your request")
        print(f"AI Error: {e}")
        return False

def run():
    speak("Hello! I'm your voice assistant. I can answer your questions. What can I do for you?")
    
    while True:
        try:
            cmd = hear()
            if cmd == "":
                continue
            if any(word in cmd for word in ['exit']):
                speak("Goodbye! Have a great day!")
                break
            else:
                handle_ai_query(cmd)
                
        except KeyboardInterrupt:
            speak("Goodbye!")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            speak("Sorry, something went wrong. Please try again.")

if __name__ == "__main__":
    print("Starting Voice Assistant...")
    print("Say 'exit' to stop")
    run()