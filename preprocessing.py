import re


def cleanup_text(texts):
    cleaned_text = []
    for text in texts:
        # remove ugly &quot and &amp
        text = re.sub(r"&quot;(.*?)&quot;", "\g<1>", text)
        text = re.sub(r"&amp;", "", text)

        # replace emoticon
        text = re.sub(
            r"(^| )(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)",
            "\g<1>TOKEMOTICON",
            text,
        )

        text = text.lower()
        text = text.replace("tokemoticon", "TOKEMOTICON")

        # replace url
        text = re.sub(
            r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?",
            "TOKURL",
            text,
        )

        # replace mention
        text = re.sub(r"@[\w]+", "TOKMENTION", text)

        # replace hashtag
        text = re.sub(r"#[\w]+", "TOKHASHTAG", text)

        # replace dollar
        text = re.sub(r"\$\d+", "TOKDOLLAR", text)

        # remove punctuation
        text = re.sub("[^a-zA-Z0-9]", " ", text)

        # remove multiple spaces
        text = re.sub(r" +", " ", text)

        # remove newline
        text = re.sub(r"\n", " ", text)

        cleaned_text.append(text)
    return cleaned_text
