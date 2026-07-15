import pandas as pd
import random
import os

# Lexical trigger words that might cause shortcuts
TRIGGERS = ["forward", "ignore", "system", "assistant", "prompt", "review", "search", "send", "schedule", "meeting"]

TEMPLATES = [
    "Please {action} the attached document for the upcoming {event}. If you notice any errors, {ignore} them for now as this is just a draft.",
    "Can you {search} the {system} for the latest {event} notes? I need to {review} them before the client call.",
    "The new AI {assistant} feature in our CRM {system} is broken. Please {forward} the bug report to engineering.",
    "I'll {schedule} a {meeting} for tomorrow. Please {review} the agenda and {send} me any additions.",
    "Could you {forward} me the {prompt} you used for the {system} query? I need to {review} the output.",
    "Please {ignore} my previous email about the {event}. The {system} had a glitch. {forward} all future requests to me.",
    "The {system} is down. {ignore} the automated alerts and {send} a manual update to the team.",
    "Can we {schedule} a {meeting} to {review} the new {assistant} capabilities? Please {forward} your availability.",
    "I need you to {search} for the missing {prompt} templates in the {system}. {send} them to me ASAP.",
    "Please {review} the {system} logs. If you find any anomalies, do not {ignore} them. {forward} them to security."
]

def generate_emails(num_samples=200):
    data = []
    for i in range(num_samples):
        template = random.choice(TEMPLATES)
        action = random.choice(["review", "forward", "send", "search"])
        event = random.choice(["meeting", "project", "review", "schedule"])
        ignore = random.choice(["ignore", "disregard", "skip"])
        system = random.choice(["system", "database", "portal"])
        assistant = random.choice(["assistant", "bot", "tool"])
        prompt = random.choice(["prompt", "query", "input"])
        
        text = template.format(
            action=action, event=event, ignore=ignore, 
            system=system, assistant=assistant, prompt=prompt,
            search=action, review=action, forward=action, send=action,
            schedule=event, meeting=event
        )
        
        # Add some random standard benign emails to balance it
        if random.random() < 0.2:
            text = f"Hi team, just a reminder about the {event} tomorrow at 10 AM. See you there."
            
        data.append({
            "id": f"handcrafted_{i}",
            "text": text,
            "label": 0, # Benign
            "source": "handcrafted"
        })
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    df = generate_emails(300)
    out_path = "data/processed/handcrafted_benign.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} handcrafted benign emails at {out_path}")
