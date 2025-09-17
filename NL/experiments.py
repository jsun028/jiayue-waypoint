import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from specs import QuerySpec

API_KEY = os.getenv("OPENAI_API_KEY")

MODEL = ChatOpenAI(
  model="gpt-4o-mini", 
  api_key='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'
)
parser = PydanticOutputParser(pydantic_object=QuerySpec)

SYSTEM = """You translate traffic scenario descriptions into a JSON spec for a query engine.
- You must return ONLY JSON conforming to the provided schema.
- Use objects 'car1','car2',... unless told otherwise.
- Use angles in degrees unless explicitly stated radians in 'TrajectorySpec.angle_rad'.
- Prefer velocity_above(2.0..5.0) for "moving", velocity_below(2.0..3.0) for "stopped".
- For "right turn", add a trajectory constraint with template='right_arc'.
- Keyframe names: k1, k2, k3, ... in temporal order.
"""

FORMAT = """{format_instructions}"""

FEWSHOT_USER = """Right-turn scenario: car1 is initially moving opposite of car2, then after ~3s, they differ by ~90°. Car1 stops then accelerates into a right arc (~135°)."""
FEWSHOT_ASSISTANT = r"""
{{
  "objects": {{
    "counts": {{"car": 2}},
    "aliases": {{
      "car1": {{"class":"car","idx":0}},
      "car2": {{"class":"car","idx":1}}
    }}
  }},
  "keyframes": [
    {{
      "name": "k1",
      "where": {{
        "op": "AND",
        "args": [
          {{
            "op":"ATOM",
            "atom":{{"type":"heading_diff_to","obj":"car1","other_obj":"car2","value":180.0,"tol":15.0}}
          }},
          {{"op":"ATOM","atom":{{"type":"velocity_above","obj":"car1","value":2.0}}}},
          {{"op":"ATOM","atom":{{"type":"velocity_above","obj":"car2","value":2.0}}}}
        ]
      }}
    }},
    {{
      "name": "k2",
      "where": {{
        "op": "AND",
        "args": [
          {{
            "op":"ATOM",
            "atom":{{"type":"heading_diff_to","obj":"car1","other_obj":"car2","value":90.0,"tol":15.0}}
          }},
          {{"op":"ATOM","atom":{{"type":"velocity_above","obj":"car1","value":2.0}}}},
          {{"op":"ATOM","atom":{{"type":"velocity_above","obj":"car2","value":2.0}}}}
        ]
      }}
    }}
  ],
  "constraints": [
    {{"kind":"always","anchor":null,"target":"k1","duration_sec":3.0,"tol":0.01}},
    {{"kind":"always","anchor":null,"target":"k2","duration_sec":3.0,"tol":0.01}},
    {{"kind":"interframe","anchor":"k1","target":"k2","time_shift":3.0,
     "comparators":[{{"type":"heading_diff","value":90.0,"tol":15.0}}]}},
    {{"kind":"trajectory","obj":"car1","start":"k1","end":"k2",
     "template":"right_arc","angle_rad":4.71238898038469,"deviation_strength":0.0}}
  ]
}}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", FORMAT),
        ("human", FEWSHOT_USER),
        ("ai", FEWSHOT_ASSISTANT),
        MessagesPlaceholder("history"),
        ("human", "{user_request}")
    ]
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | MODEL | parser