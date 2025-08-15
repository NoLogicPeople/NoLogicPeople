from vector_index import VectorIndex


def main() -> None:
    # Build IVF index with tuned params
    index = VectorIndex(
        csv_path='scraped_items_v2.csv',
        index_path='faiss_ivf.index',
        index_type='ivf',
        nlist=1024,
        nprobe=10,
    )
    index.ensure_built(force_rebuild=True)
    print('IVF vector index built successfully.')


if __name__ == '__main__':
    main()
