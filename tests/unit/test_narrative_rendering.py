from html.parser import HTMLParser

from scripts.narrative_rendering import markdown_to_html


class RenderedHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, attrs))

    def handle_data(self, data):
        self.text.append(data)


def test_untrusted_narrative_is_text_not_active_html():
    rendered = RenderedHTML()
    rendered.feed(markdown_to_html('**Observation**\n<script>alert(1)</script>\n<img src=x onerror=alert(2)>'))
    assert not {tag for tag, _ in rendered.tags} & {"script", "img"}
    assert all(name not in {"src", "onerror", "onclick"}
               for _, attrs in rendered.tags for name, _ in attrs)
    assert "<script>alert(1)</script>" in "".join(rendered.text)
    assert "Observation" in "".join(rendered.text)
