from src.utils.prepare_mt_lowercase_data import create_mt_lowercase_data


def main():
    create_mt_lowercase_data(
        lang1="en",
        lang2="de",
        input_dir="fairseq/examples/homograph_translation/orig/en_de_standard",
        output_dir="fairseq/examples/homograph_translation/orig/en_de_lowercase",
        stats_path="fairseq/examples/homograph_translation/orig/en_de_lowercase/lowercase_stats.json",
    )


if __name__ == "__main__":
    main()