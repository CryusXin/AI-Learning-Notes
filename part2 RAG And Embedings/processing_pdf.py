# pip install pdfminer.six
# pip install nltk

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from nltk.tokenize import sent_tokenize


class PDFProcessor:
    def extract_text_from_pdf(self,filename, page_numbers=None, min_line_length=1):
        '''从 PDF 文件中（按指定页码）提取文字'''
        paragraphs = []
        buffer = ''
        full_text = ''
        # 提取全部文本
        for i, page_layout in enumerate(extract_pages(filename)):
            # 如果指定了页码范围，跳过范围外的页
            if page_numbers is not None and i not in page_numbers:
                continue
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    full_text += element.get_text() + '\n'
        # 按空行分隔，将文本重新组织成段落
        lines = full_text.split('\n')
        for text in lines:
            if len(text) >= min_line_length:
                buffer += (' ' + text) if not text.endswith('-') else text.strip('-')
            elif buffer:
                paragraphs.append(buffer)
                buffer = ''
        if buffer:
            paragraphs.append(buffer)
        return paragraphs

    def split_text(self,paragraphs, chunk_size=300, overlap_size=100, min_line_length=10):
        '''按指定 chunk_size 和 overlap_size 交叠割文本'''
        sentences = [s.strip() for p in paragraphs for s in sent_tokenize(p)]
        chunks = []
        i = 0
        while i < len(sentences):
            chunk = sentences[i]
            overlap = ''
            prev_len = 0
            prev = i - 1
            # 向前计算重叠部分
            while prev >= 0 and len(sentences[prev]) + len(overlap) <= overlap_size:
                overlap = sentences[prev] + ' ' + overlap
                prev -= 1
            chunk = overlap + chunk
            next = i + 1
            # 向后计算当前chunk
            while next < len(sentences) and len(sentences[next]) + len(chunk) <= chunk_size:
                chunk = chunk + ' ' + sentences[next]
                next += 1
            chunks.append(chunk)
            i = next
        return chunks

