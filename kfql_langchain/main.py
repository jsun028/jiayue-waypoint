from full_chain import build_full_chain
from ir_chain import prompt # for debug

"""
* Compiler Flow

1. Parsing:                 NL → JSON IR (llm, verified by pydantic schema, and static type checking)
2. Semantic Analysis:       core relational operators, built-in functions, and UDFs
3. Code Generation:         IR → DSL (compile_to_kfql, a deterministic transformation)
4. Execution/Optimization:  actual query execution (TODO)


opportunities:
- static type checking: give feedback to the llm
- semantic analysis:  give feedback to the llm 
    (search available built-in functions and UDFs, and if not, give feedback to the llm to generate new functions)

- code generation: TBD
- execution/optimization: TBD
- feedback to the user: TBD

"""

QUERY = "two cars moving in opposite directions, \
and then in perpendicular ways after crossing the intersection, \
after 3 seconds"

def run():
    chain = build_full_chain(debug=True)

    formatted = prompt.format_prompt(nl_query=QUERY)
    print("===== Formatted Prompt (debug) =====")
    print(formatted.to_string())
    ##
    
    result = chain.invoke({"nl_query": QUERY})
    print("===== Final DSL Query =====\n")
    print(result["dsl"])

if __name__ == "__main__":
    run()
