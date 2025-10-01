from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from agents.exceptions import InputGuardrailTripwireTriggered
from pydantic import BaseModel
import asyncio
import aiosqlite
from dotenv import load_dotenv

load_dotenv()

# -------------------
# Output schemas
# -------------------
class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

class AccessControlOutput(BaseModel):
    allowed: bool
    reasoning: str

class ClassificationOutput(BaseModel):
    subject: str  # must be "math" or "history"

# -------------------
# Agents
# -------------------
guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework. Respond with is_homework true/false and reasoning.",
    output_type=HomeworkOutput,
)

classifier_agent = Agent(
    name="Classifier Agent",
    instructions=(
        "Classify the homework question as either 'math' or 'history'. "
        "Respond with the subject field as exactly 'math' or 'history'."
    ),
    output_type=ClassificationOutput,
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide the final numeric answer only.",
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)

access_control_agent = Agent(
    name="Access Control Agent",
    instructions=(
        "You are an access control checker. "
        "You are given a natural-language policy, the student's subject and age, "
        "and the type of question (math or history). "
        "Respond ONLY with allowed true/false and reasoning."
    ),
    output_type=AccessControlOutput,
)

# -------------------
# SQLite helpers
# -------------------
DB_PATH = "students.db"

async def get_student(student_name: str):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT subject, age FROM students WHERE name = ?", (student_name,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return {"subject": row[0], "age": row[1]}
            raise ValueError(f"No student found with name {student_name}")

# -------------------
# Guardrail
# -------------------
async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)

    # Block if not homework
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework,
    )

# -------------------
# Runner helper
# -------------------
async def ask_question(student_name, question, policy):
    try:
        student = await get_student(student_name)

        # Step 1: Guardrail check
        gr_result = await Runner.run(guardrail_agent, question)
        gr_output = gr_result.final_output_as(HomeworkOutput)

        if not gr_output.is_homework:
            print(f"BLOCKED: {gr_output.reasoning}")
            return
            
        # Step 2: Classification
        classifier_result = await Runner.run(
            classifier_agent,
            question,
            context={
                "student_subject": student["subject"],
                "student_age": student["age"],
            },
        )
        classification = classifier_result.final_output_as(ClassificationOutput)
        question_subject = classification.subject.lower()

        # Step 3: Access Control
        ac_result = await Runner.run(
            access_control_agent,
            f"Policy: {policy}\nStudent subject: {student['subject']}\n"
            f"Student age: {student['age']}\nQuestion subject: {question_subject}",
        )
        ac_output = ac_result.final_output_as(AccessControlOutput)

        if not ac_output.allowed:
            print(f"ACCESS DENIED: {ac_output.reasoning}")
            return

        # Step 4: Tutor (only if allowed)
        tutor = math_tutor_agent if question_subject == "math" else history_tutor_agent
        tutor_result = await Runner.run(
            tutor,
            question,
            context={"student_subject": student["subject"], "student_age": student["age"]},
        )

        print(f"{student_name} (age {student['age']}) asked: {question}")
        print("Answer:", tutor_result.final_output)

    except InputGuardrailTripwireTriggered:
        print(f"{student_name} is not allowed to ask: {question}")
    except ValueError as e:
        print(e)

# -------------------
# Main
# -------------------
async def main():
    policy = input("Define access control policy (in English): ").strip()
    student_name = input("Enter student name (alice or bob): ").strip().lower()
    question = input("Enter your question: ").strip()

    await ask_question(student_name, question, policy)

if __name__ == "__main__":
    asyncio.run(main())
