"""ACE prompts, ported to InterPhyre.

Sources (github.com/ace-agent/ace @ 82709de and the paper's appendix F):

  GENERATOR_BLOCK   AppWorld generator prompt (paper Fig. 9): the playbook sits
                    between PLAYBOOK_BEGIN/PLAYBOOK_END with "read the playbook
                    first" and key instruction 9 ("treat the cheatsheet as a
                    tool").  The bullet-citation line is the ReAct form of the
                    FiNER generator's `bullet_ids` output field (Fig. 12): the
                    Reflector can only tag bullets the Generator says it used.
  REFLECTOR_PROMPT  `ace/prompts/reflector.py:REFLECTOR_PROMPT_NO_GT`, with the
                    "reasoning trace" made a Thought/Action/Observation
                    trajectory and the AppWorld reflector's grounding line
                    ("grounded in execution feedback") added.  InterPhyre has no
                    reference answer, so only the no-ground-truth variant
                    exists here -- the simulator verdict is the feedback, as in
                    the paper's label-free AppWorld setting.
  CURATOR_PROMPT    `ace/prompts/curator.py:CURATOR_PROMPT_NO_GT`, verbatim apart
                    from the task-context label.

What is deliberately NOT added: any physics strategy, worked solution or
level-specific hint.  Those would be scaffolding the method does not have, and
they would leak the insight the level is meant to test.
"""

GENERATOR_BLOCK = """## ACE Playbook
You are also provided with a curated playbook of strategies, insights, common mistakes, and proven solutions accumulated from your own previous attempts at this kind of puzzle. Read the playbook first, then solve the task by explicitly leveraging each relevant section:

PLAYBOOK_BEGIN
{playbook}
PLAYBOOK_END

Treat the playbook as a tool. Use only the parts that are relevant and applicable to your specific situation and task context, otherwise use your own judgement.
Each playbook line starts with a bullet ID such as [sai-00001]. Whenever a bullet informs a step, cite its ID in that step's Thought (e.g. "Following [sai-00001], ..."), so it can be credited or corrected afterwards."""


REFLECTOR_PROMPT = """You are an expert analyst and educator. Your job is to diagnose the current trajectory of an agent solving a 2D physics placement puzzle: identify what went wrong (or could be better), grounded in the simulator feedback.

**Instructions:**
- Carefully analyze the agent's trajectory (its Thoughts, Actions, and the simulator Observations) to identify where it went wrong
- Take the environment feedback into account
- Identify specific conceptual errors, calculation mistakes, or misapplied strategies
- Provide actionable insights that could help the agent avoid this mistake in the future
- Focus on the root cause, not just surface-level errors
- Be specific about what the agent should have done differently
- You will receive bulletpoints that are part of playbook that's used by the agent to solve the task.
- You need to analyze these bulletpoints, and give the tag for each bulletpoint, tag can be ['helpful', 'harmful', 'neutral'] (for the agent to solve the task)

Your output should be a json object, which contains the following fields
  - reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
  - error_identification: what specifically went wrong in the reasoning?
  - root_cause_analysis: why did this error occur? What concept was misunderstood?
  - correct_approach: what should the agent have done instead?
  - key_insight: what strategy, formula, or principle should be remembered to avoid this error?
  - bullet_tags: a list of json objects with bullet_id and tag for each bulletpoint used by the agent




**Task:**
{question}

**Agent's Trajectory:**
{trajectory}

**Agent's Final Placement:**
{prediction}

**Environment Feedback:**
{feedback}

**Part of Playbook that's used by the agent to solve the task:**
{bullets_used}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",
  "error_identification": "[What specifically went wrong in the reasoning?]",
  "root_cause_analysis": "[Why did this error occur? What concept was misunderstood?]",
  "correct_approach": "[What should the agent have done instead?]",
  "key_insight": "[What strategy, formula, or principle should be remembered to avoid this error?]",
  "bullet_tags": [
    {{"id": "calc-00001", "tag": "helpful"}},
    {{"id": "fin-00002", "tag": "harmful"}}
  ]
}}

---
"""


CURATOR_PROMPT = """You are a master curator of knowledge. Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous attempt.

**Context:**
- The playbook you created will be used to help answering similar questions.
- The reflection is generated using environment feedback that will NOT be available when the playbook is being used.

**CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.**

**Instructions:**
- Review the existing playbook and the reflection from the previous attempt
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook
- Avoid redundancy - if similar advice already exists, only add new content that is a perfect complement to the existing playbook
- Do NOT regenerate the entire playbook - only provide the additions needed
- Focus on quality over quantity - a focused, well-organized playbook is better than an exhaustive one
- Format your response as a PURE JSON object with specific sections
- For any operation if no new content to add, return an empty list for the operations field
- Be concise and specific - each addition should be actionable


**Training Context:**
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}

**Current Playbook Stats:**
{playbook_stats}

**Recent Reflection:**
{recent_reflection}

**Current Playbook:**
{current_playbook}

**Task Context:**
{question_context}

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- operations: a list of operations to be performed on the playbook
  - type: the type of operation to be performed
  - section: the section to add the bullet to
  - content: the new content of the bullet

**Available Operations:**
1. ADD: Create new bullet points with fresh IDs
    - section: the section to add the new bullet to
    - content: the new content of the bullet. Note: no need to include the bullet_id in the content like '[ctx-00263] helpful=1 harmful=0 ::', the bullet_id will be added by the system.

**RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations here]",
  "operations": [
    {{
      "type": "ADD",
      "section": "formulas_and_calculations",
      "content": "[New calculation method...]"
    }}
  ]
}}

---
"""

#: `ace/ace.py:ACE._initialize_empty_playbook`, verbatim.  The sections are the
#: reference implementation's generic defaults; no physics-specific sections are
#: pre-seeded, so every bullet in a run was written by the Curator.
EMPTY_PLAYBOOK = """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""

#: Feedback strings the reference loop hands the Reflector
#: (`ace/ace.py:515,558`), adapted from "matches ground truth" to the verdict.
FEEDBACK_SUCCESS = "The simulator verified this placement as a SUCCESS."
FEEDBACK_FAILURE = "The agent did NOT solve the puzzle."
