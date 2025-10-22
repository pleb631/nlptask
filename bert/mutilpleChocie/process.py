from datasets import load_dataset

ds = load_dataset("allenai/swag", "regular")

ds = ds.remove_columns(["video-id","fold-ind","sent2","sent1","gold-source"])

ds = ds.filter(
        lambda x: x["startphrase"] is not None
        and x["ending0"] is not None
        and x["ending1"] is not None
        and x["ending2"] is not None
        and x["ending3"] is not None
        and x["label"] is not None
    )
ds.save_to_disk("./data/")