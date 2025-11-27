import json
import uuid
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents import RoomInputOptions, MetricsCollectedEvent
from livekit.plugins import murf, silero, deepgram, google, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")

FRAUD_DB_FILE = "shared-data/fraud_cases.json"


# ---------- DB HELPERS ----------
def load_fraud_cases():
    with open(FRAUD_DB_FILE, "r") as f:
        return json.load(f)


def save_fraud_cases(cases):
    with open(FRAUD_DB_FILE, "w") as f:
        json.dump(cases, f, indent=2)


# -------------- FRAUD AGENT --------------
class FraudAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a calm, professional fraud verification agent for a fictional bank. "
                        "Never ask for full card numbers, passwords, or sensitive credentials."
        )
        self.active_case = None
        self.stage = "ask_username"  # call flow stages

    async def on_start(self, session: AgentSession):
        await session.say(
            "Hello, this is the Fraud Prevention Department from SafeBank. "
            "We detected a suspicious transaction and need to verify a few details. "
            "May I have your name to verify the case?",
            allow_interruptions=True,
        )

    # ---------- LOAD CASE ----------
    def find_case(self, name):
        cases = load_fraud_cases()
        for case in cases:
            if case["userName"].lower() == name.lower():
                return case
        return None

    # ---------- MAIN LOGIC ----------
    async def on_response(self, response, session: AgentSession):
        msg = response.text.strip().lower()

        # --- Step 1: Get username ---
        if self.stage == "ask_username":
            self.active_case = self.find_case(msg)
            if not self.active_case:
                await session.say("I could not find any fraud cases under that name. Please try again.")
                return

            self.stage = "verification"
            await session.say(
                f"Thank you. Before we proceed, please answer this verification question: "
                f"{self.active_case['securityQuestion']}"
            )
            return

        # --- Step 2: Verify security question ---
        if self.stage == "verification":
            if msg != self.active_case["securityAnswer"].lower():
                self.active_case["status"] = "verification_failed"
                save_fraud_cases(load_fraud_cases())
                await session.say(
                    "I’m sorry, that didn’t match our records. For security reasons, I cannot continue this verification."
                )
                return

            self.stage = "transaction"
            await session.say(
                f"Thank you for verifying. Here are the details of the suspicious transaction: "
                f"A charge of {self.active_case['amount']} at {self.active_case['merchant']} "
                f"in {self.active_case['location']} on {self.active_case['timestamp']}. "
                f"The card used ends with {self.active_case['cardEnding']}. "
                f"Did you make this transaction?"
            )
            return

        # --- Step 3: Ask if transaction is legitimate ---
        if self.stage == "transaction":
            cases = load_fraud_cases()

            if "yes" in msg:
                self.active_case["status"] = "confirmed_safe"
                await session.say("Thank you. We have marked this transaction as legitimate.")
            else:
                self.active_case["status"] = "confirmed_fraud"
                await session.say(
                    "Thanks for confirming. We have blocked your card and started a fraud dispute case. "
                    "A new card will be issued shortly."
                )

            # Save updated case
            for i, c in enumerate(cases):
                if c["userName"] == self.active_case["userName"]:
                    cases[i] = self.active_case
            save_fraud_cases(cases)

            await session.say("Your case has been updated. Thank you for your time. Stay safe!")
            self.stage = "done"
            return


# ---------- PREWARM ----------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# ---------- ENTRYPOINT ----------
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(),
        tts=murf.TTS(voice="Matthew", style="Conversation"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _collect(ev: MetricsCollectedEvent):
        usage.collect(ev.metrics)

    await session.start(
        agent=FraudAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
