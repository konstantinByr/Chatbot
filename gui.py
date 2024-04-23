import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox


class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        master.title("Chatbot")

        # Textbereich für Chat-Verlauf
        self.chat_history = scrolledtext.ScrolledText(master, state='disabled', height=15, width=50)
        self.chat_history.grid(row=0, column=0, columnspan=2)

        # Eingabefeld für Benutzernachrichten
        self.msg_entry = tk.Entry(master, width=40)
        self.msg_entry.grid(row=1, column=0)

        # Senden-Button
        self.send_button = tk.Button(master, text="Senden", command=self.send_message)
        self.send_button.grid(row=1, column=1)

    def send_message(self):
        user_input = self.msg_entry.get()
        self.msg_entry.delete(0, tk.END)
        self.update_chat_history("Du: " + user_input)

        # Integriere die Chatbot-Antwortlogik hier
        response = self.get_response(user_input)
        self.update_chat_history("Bot: " + response)

    def update_chat_history(self, message):
        self.chat_history.config(state='normal')
        self.chat_history.insert(tk.END, message + "\n")
        self.chat_history.yview(tk.END)
        self.chat_history.config(state='disabled')

    def get_response(self, user_input):
        # Platziere die Logik deines Chatbots hier
        return "Antwort auf '" + user_input + "'"

if __name__ == "__main__":
    root = tk.Tk()
    gui = ChatbotGUI(root)
    root.mainloop()