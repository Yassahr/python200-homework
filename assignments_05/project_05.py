from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    is_it_safe= not flagged
    if is_it_safe:
        return is_it_safe
    else:
        print('This was flagged for inappropiate content')

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        print("JSON Invalid")
        print(response.choices[0].message.content)
    return response.choices[0].message.content
    


system_prompt= '''
You are an expert job application coach helping early-to-mid career tech/technology professionals 
improve their resumes, cover letters, and LinkedIn profiles to land more interviews.

Your role:
- Review, rewrite, and give specific feedback on job application materials
- Tailor content to match job descriptions when provided
- Suggest stronger action verbs, quantifiable achievements, and clearer formatting
- Help the user present their experience confidently and authentically
'''
# I made sure I added these in bullet point to seperate concepts but I also added information suggestions I hear from career coaching sites


def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list.Do not wrap it in markdown code fences or add any explanation. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```

    Format your response as:
    Original Bullet Point: <original bullet point>
    Improved Bullet Point: <improved bullet point>
    """
    
    messages = [{"role": "user", "content": prompt}]
    return messages


bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

improvement=get_completion(rewrite_bullets(bullets))




def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    return messages

job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."
cover_letter=get_completion(generate_cover_letter(job_title, background))

#They were very descriptive and did a good job at refocusing transferable skills in their previous careers and recentering to the skills needed for the current job position


def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    exchange_count = 0
    max_exchanges = 3
    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while exchange_count < max_exchanges:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue 

        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
        
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            
            improved_bullets=get_completion(rewrite_bullets(raw_bullets))
            messages.append({"role": "user", "content": user_input})

            print(improved_bullets)

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            # YOUR CODE: call generate_cover_letter() and print the result
            cover_letter=get_completion(generate_cover_letter(job_title, background))       
            print(cover_letter) 
        # 7. Otherwise, handle it as a regular chat turn
        else:
            # YOUR CODE:
            message= user_input
            reply_free_text = get_completion([{"role": "user", "content": user_input}])
            print(reply_free_text)
            messages.append({"role": "assistant", "content": reply_free_text})
            pass
        exchange_count += 1
        print(len(messages))
        if exchange_count >= max_exchanges:
            print('Max entries, start new session')


if __name__ == "__main__":
    run_chatbot()



    # 2 with all AI it runs the risk of hallunicating which could lead to false, incorrect, or nonsensical informations be put onto your resume
    # 3 I would add a disclaimer, usage limit, and add additional logic to make sure the AI's responses does not go beyond the scope of the system support role