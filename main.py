import sys
import os
import asyncio
import argparse

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from src.pipeline import EnterpriseRAGPipeline
from src.common.config import settings

def print_header():
    print("=" * 65)
    print("  ENTERPRISE AGENTIC RAG SYSTEM - REAL-TIME ENGINE")
    print("=" * 65)

def cmd_ingest(args, pipeline: EnterpriseRAGPipeline):
    target_path = args.path
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]

    if not os.path.exists(target_path):
        print(f"❌ Error: Path does not exist: {target_path}")
        return

    print(f"\n📂 Ingesting from: {target_path}")
    if os.path.isdir(target_path):
        results = pipeline.ingest_directory(target_path, allowed_roles=roles)
        print(f"✅ Ingested {len(results)} document(s) from directory.")
        for r in results:
            print(f"   • {r['title']}: {r['child_chunks_count']} chunks, {r['total_indexed_nodes']} graph nodes")
    else:
        result = pipeline.ingest_file(target_path, allowed_roles=roles)
        print(f"✅ Ingested file '{result['title']}':")
        print(f"   • Parent Chunks: {result['parent_chunks_count']}")
        print(f"   • Child Chunks: {result['child_chunks_count']}")
        print(f"   • Knowledge Graph Nodes: {result['total_indexed_nodes']}")
        print(f"   • Knowledge Graph Edges: {result['total_indexed_edges']}")

    print(f"💾 Persistent state saved to: {pipeline.storage_dir}\n")

def setup_agent_event_logger(pipeline: EnterpriseRAGPipeline, verbose: bool = True):
    if not verbose:
        return

    def on_event(event):
        e_type = event.event_type
        d = event.data
        if e_type == "ON_PLAN":
            print(f"\n📋 [Agent Plan: {len(d.get('subgoals', []))} Sub-goals]")
            for idx, g in enumerate(d.get("subgoals", [])):
                print(f"   {idx+1}. {g}")
        elif e_type == "ON_THOUGHT":
            print(f"  🤔 Thought (Step {d.get('step')}): {d.get('thought')}")
        elif e_type == "ON_ACTION":
            print(f"  ⚙️ Action: calling tool `{d.get('tool')}`({d.get('args')})")
        elif e_type == "ON_OBSERVATION":
            rel_badge = "✓ Relevant" if d.get("is_relevant") else "⚠️ Low relevance"
            obs_preview = str(d.get("observation", ""))[:120].replace('\n', ' ')
            print(f"  📥 Observation [{rel_badge}]: {obs_preview}...")

    pipeline.event_bus.subscribe(on_event)

async def cmd_query(args, pipeline: EnterpriseRAGPipeline):
    query_str = args.query
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]

    provider_name = pipeline.llm_provider.provider_type.upper()
    model_name = pipeline.llm_provider.model
    print(f"\n🧠 Agent Engine: [{provider_name} : {model_name}]")
    print(f"🔎 Query: '{query_str}'")

    setup_agent_event_logger(pipeline, verbose=True)
    response = await pipeline.query(query_str, user_roles=roles)

    status_badge = "✅ VERIFIED" if response.verification_passed else ("⚠️ FALLBACK / ABSTAINED" if response.fallback_used else "❌ REJECTED")
    print(f"\n   Status: {status_badge}")
    print(f"   Faithfulness: {response.faithfulness_score:.2f} | Latency: {response.latency_ms:.1f}ms | Retries: {response.retries_used}")
    if response.citations:
        print(f"   Citations: {', '.join(response.citations)}")
    
    clean_ans = response.answer.replace('\r', '')
    print(f"\n💬 Answer:\n{clean_ans}\n")

async def cmd_chat(args, pipeline: EnterpriseRAGPipeline):
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    provider_name = pipeline.llm_provider.provider_type.upper()
    model_name = pipeline.llm_provider.model

    print_header()
    print(f"Autonomous Real-Time Agentic Chat [Engine: {provider_name} : {model_name}]")
    print(f"Active Chunks: {len(pipeline.chunk_store)} | Graph Nodes: {len(pipeline.graph_indexer.graph.nodes)}")
    print("Type your questions below. Enter 'exit' or 'quit' to exit.\n")

    setup_agent_event_logger(pipeline, verbose=True)

    while True:
        try:
            user_input = input("\nUser ❯ ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting chat.")
                break

            response = await pipeline.query(user_input, user_roles=roles)
            status_badge = "✅ VERIFIED" if response.verification_passed else ("⚠️ ABSTAINED" if response.fallback_used else "❌ REJECTED")
            
            clean_ans = response.answer.replace('\r', '')
            print(f"\nAgent [{status_badge} | {response.latency_ms:.1f}ms | Faithfulness: {response.faithfulness_score:.2f}]:")
            print(f"{clean_ans}")
            if response.citations:
                print(f"Citations: {', '.join(response.citations)}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break


def cmd_status(args, pipeline: EnterpriseRAGPipeline):
    print_header()
    print(f"Storage Directory: {pipeline.storage_dir}")
    print(f"Indexed Parent Chunks: {len(pipeline.parent_store)}")
    print(f"Indexed Child Chunks:  {len(pipeline.chunk_store)}")
    print(f"Knowledge Graph Nodes: {len(pipeline.graph_indexer.graph.nodes)}")
    print(f"Knowledge Graph Edges: {len(pipeline.graph_indexer.graph.edges)}")

def cmd_serve(args):
    import uvicorn
    print_header()
    print(f"Starting Production FastAPI Server on port {args.port}...")
    uvicorn.run("src.api.server:app", host=args.host, port=args.port, reload=args.reload)

def main():
    parser = argparse.ArgumentParser(description="Enterprise Agentic RAG CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a real document or directory")
    ingest_parser.add_argument("path", help="Path to file (.pdf, .md, .txt) or directory")
    ingest_parser.add_argument("--roles", default="public", help="Comma-separated allowed roles (e.g. public,employee)")

    # Query command
    query_parser = subparsers.add_parser("query", help="Run a single real-time query")
    query_parser.add_argument("query", help="Question text")
    query_parser.add_argument("--roles", default="public", help="Comma-separated user roles")

    # Interactive Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive real-time chat session")
    chat_parser.add_argument("--roles", default="public", help="Comma-separated user roles")

    # Status command
    subparsers.add_parser("status", help="Show index and knowledge graph statistics")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI HTTP REST server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host interface")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number")
    serve_parser.add_argument("--reload", action="store_true", help="Auto-reload on code change")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "serve":
        cmd_serve(args)
        return

    pipeline = EnterpriseRAGPipeline(storage_dir="data/storage")


    if args.command == "ingest":
        cmd_ingest(args, pipeline)
    elif args.command == "query":
        asyncio.run(cmd_query(args, pipeline))
    elif args.command == "chat":
        asyncio.run(cmd_chat(args, pipeline))
    elif args.command == "status":
        cmd_status(args, pipeline)

if __name__ == "__main__":
    main()
