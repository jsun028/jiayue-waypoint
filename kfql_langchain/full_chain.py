from langchain.chains import SequentialChain

from ir_chain import ir_chain                       # Step 1: NL → JSON IR
from transform_chain import IRTransformChain        # Step 2~3: JSON IR → DSL
from udf_resolver import UDFResolver
from resolver_chain import UDFResolveChain


# dummy embedding function (actual: OpenAI embeddings)
dummy_emb = lambda text: [float(ord(c)) % 10 for c in text][:10]

def build_full_chain(debug=False):
    
    # resolver = UDFResolver(embedding_fn=dummy_emb, threshold=0.75)
    # resolver.register_udf("perpendicular", "two objects moving at 90 degrees")

    # resolve_chain = UDFResolveChain(resolver)
    transform_chain = IRTransformChain()

    if debug:
        def debug_wrapper(inputs):
            ir_result = ir_chain.invoke(inputs)
            print("===== IRChain Output =====\n")
            # IRSchema (Pydantic) object, so use .model_dump_json
            print(ir_result.model_dump_json(indent=2))
            return ir_result

        from langchain.schema.runnable import RunnableLambda
        ir_with_debug = RunnableLambda(debug_wrapper)

        # debug mode also pass through resolver and transform
        # return ir_with_debug | resolve_chain | transform_chain
        return ir_with_debug | transform_chain
    else:
        # return ir_chain | resolve_chain | transform_chain
        return ir_chain | transform_chain

