from .encode import main

if __name__ == "__main__":
    main(
        encoded_name="topic_title.pkl",
        dataset_name="neuclir/csl-topics",
        encode_is_qry=True,
        query_field="topic_title",
        use_passage_enc_for_query=False,
        # dataset_proc_num=None,
        q_max_len=32,
    )

    main(
        encoded_name="topic_description.pkl",
        dataset_name="neuclir/csl-topics",
        encode_is_qry=True,
        query_field="topic_description",
        # dataset_proc_num=None,
        use_passage_enc_for_query=False,
        q_max_len=128,
    )

    main(
        encoded_name="topic_narrative.pkl",
        dataset_name="neuclir/csl-topics",
        encode_is_qry=True,
        query_field="topic_narrative",
        # dataset_proc_num=None,
        use_passage_enc_for_query=True,
        q_max_len=512,
    )
