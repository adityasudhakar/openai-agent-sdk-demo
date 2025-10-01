from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from agents.exceptions import InputGuardrailTripwireTriggered
from pydantic import BaseModel
import asyncio
import aiosqlite
from dotenv import load_dotenv
import os

load_dotenv()

# -------------------
# Output schema
# -------------------
class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

# -------------------
# Agents
# -------------------
guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework. Respond with is_homework true/false and reasoning.",
    output_type=HomeworkOutput,
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
# Triage Agent
# -------------------
triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine whether the homework question is history or math, and hand it off to the correct tutor.",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[InputGuardrail(guardrail_function=homework_guardrail)],
)

# -------------------
# Runner helper
# -------------------
async def ask_question(student_name, question):
    try:
        student = await get_student(student_name)
        result = await Runner.run(
            triage_agent,
            question,
            context={"student_subject": student["subject"], "student_age": student["age"]},
        )
        print(f"{student_name} (age {student['age']}) asked: {question}")
        print("Answer:", result.final_output)
    except InputGuardrailTripwireTriggered:
        print(f"{student_name} is not allowed to ask: {question}")
    except ValueError as e:
        print(e)

# -------------------
# Main
# -------------------
async def main():
    student_name = input("Enter student name (alice or bob): ").strip().lower()
    question = input("Enter your question: ").strip()
    await ask_question(student_name, question)

if __name__ == "__main__":
    asyncio.run(main())
