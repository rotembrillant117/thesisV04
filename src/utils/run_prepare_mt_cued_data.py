from src.utils.prepare_mt_cued_data import create_mt_cued_data


def main():
    create_mt_cued_data(
        lang1="en",
        lang2="de",
        input_dir="fairseq/examples/homograph_translation/orig/en_de_standard",
        output_dir="fairseq/examples/homograph_translation/orig/en_de_homograph_marked",
        stats_path="fairseq/examples/homograph_translation/orig/en_de_homograph_marked/homograph_stats.json",
        min_count=3,
    )


if __name__ == "__main__":
    main()