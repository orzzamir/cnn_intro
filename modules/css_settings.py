from IPython.core.display import HTML


def get_settings():
    str = """
    <style>
    figure {
        display:block;
        margin: 20px; /* adjust as needed */
        text-align: center;
    }
    figure img {
        vertical-align: center;

    }
    figure figcaption {
        text-align: center;
    }
    .centerImage
    {
        text-align:center;
        display:block;
    }
    </style>
    """
    return str

