import click
from ingestion.pdf_reader import extract_pages
from index.index_builder import IndexBuilder
from index.searcher import Searcher
from agent.researcher_agent import ResearchAssistantAgent


@click.group()
def cli():
    pass


@cli.command()
@click.argument("pdf_folder")
@click.argument("index_dir")
def build_index(pdf_folder, index_dir):
    docs = []
    for f in os.listdir(pdf_folder):
        if f.lower().endswith(".pdf"):
            docs.extend(extract_pages(os.path.join(pdf_folder, f)))

    builder = IndexBuilder(index_dir)
    builder.build(docs)
    click.echo("Index built successfully.")


@cli.command()
@click.argument("query")
@click.argument("index_dir")
def ask(query, index_dir):
    searcher = Searcher(index_dir)
    # user must load PDFs separately.. (or store text in metadata if desired)
    agent = ResearchAssistantAgent()
    hits = searcher.search(query, k=5)
    answer = agent.answer(query, hits, documents=None)
    click.echo(answer)


if __name__ == "__main__":
    cli()
