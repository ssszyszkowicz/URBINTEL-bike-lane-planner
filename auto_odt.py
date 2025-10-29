from odf.opendocument import OpenDocumentText, OpenDocumentDrawing
from odf.style import Style, TextProperties, GraphicProperties, PageLayout, MasterPage, Header, Footer, FontFace, ParagraphProperties, TableColumnProperties, TableCellProperties, BackgroundImage
from odf.text import H, P, Span, List, ListItem, ListStyle, ListLevelStyleNumber, ListLevelStyleBullet
from odf.draw import Frame, Image, Page, Object, TextBox
from odf import easyliststyle
from odf.table import Table, TableColumn, TableRow, TableCell
import copy

# defaults
style_types = {
    'b': {
        'fontweight': 'bold',
        'fontweightasian': "bold",
        'fontweightcomplex': "bold"
    },
    'i': {
        'fontstyle': "italic",
        'fontstyleasian': "italic",
        'fontstylecomplex': "italic"
    },
    'u': {
        'textunderlinestyle': "solid",
        'textunderlinewidth': "auto",
        'textunderlinecolor': "font-color"
    }
}

fontsize_dict = {'': '12pt', '0': '14pt', '1': '16pt', '2': '18pt'}


class OdtDocument:
    def __init__(self, *args):
        self.doc = OpenDocumentText()

        # styles
        self.default_text_style = self.createTextStyle("")

        self.default_font_face = FontFace(name="Arial",
                                          fontfamily="Arial",
                                          fontfamilygeneric="swiss",
                                          fontpitch="variable")

        self.doc.fontfacedecls.addElement(self.default_font_face)

    def save(self, filepath):
        self.doc.save(filepath)

    def splitStyleString(self, txt):
        if '#' in txt:
            first, sec = txt.split('#')
        else:
            first, sec = txt, ''

        return first.replace('*', ''), sec

    def createTextStyle(self, txt, fontname=''):
        stl = None
        try:
            stl = self.doc.getStyleByName(txt + "_" + fontname)
        except:
            arg = {}

            if 'b' in txt:
                arg = {**arg, **style_types['b']}
            if 'i' in txt:
                arg = {**arg, **style_types['i']}
            if 'u' in txt:
                arg = {**arg, **style_types['u']}

            fs = {
                'fontsize': (fontsize_dict[(re_findall(r'\d+', txt)
                                            or [''])[0]])
            }

            arg = {**arg, **fs}

            stl = Style(name=txt + "_" + fontname, family="text")
            stl.addElement(TextProperties(fontname=fontname, **arg))
            self.doc.automaticstyles.addElement(stl)

        return stl

    def createTableStyle(self, txt):
        stl = None
        try:
            stl = self.doc.getStyleByName(txt)
        except:
            bg_color = 'transperant' if txt == '' else "#" + txt

            stl = Style(name=txt, family="table-cell")
            tcp = TableCellProperties(backgroundcolor=bg_color)
            tcp.addElement(BackgroundImage())
            stl.addElement(tcp)
            self.doc.automaticstyles.addElement(stl)

        return stl

    def addTableToDocument(self,
                           itemList,
                           column_number=3,
                           text_font_size='12pt'):

        mixedListSpec = u'1.!\u273f!a)'

        listStyle = easyliststyle.styleFromString('mix1', mixedListSpec, '!',
                                                  '0.8cm',
                                                  easyliststyle.SHOW_ONE_LEVEL)
        self.doc.styles.addElement(listStyle)

        centerstl = Style(name='centerstl', family="paragraph")
        centerstl.addElement(
            ParagraphProperties(textalign="center", justifysingleword="false"))
        self.doc.automaticstyles.addElement(centerstl)

        widthshort = Style(name="Wshort", family="table-column")
        widthshort.addElement(TableColumnProperties(columnwidth="1.7cm"))
        self.doc.automaticstyles.addElement(widthshort)

        widthwide = Style(name="Wwide", family="table-column")
        widthwide.addElement(TableColumnProperties(columnwidth="1.5in"))
        self.doc.automaticstyles.addElement(widthwide)

        listItem = ListItem()
        listArray = List()

        table = Table()
        table.addElement(
            TableColumn(numbercolumnsrepeated=column_number,
                        stylename=widthwide))
        #table.addElement(TableColumn(numbercolumnsrepeated=3,stylename=widthwide))

        listArray.setAttribute('stylename', 'mix1')

        para = None

        n = 0

        lines = [
            list(filter(None, l.split('`~')))
            for l in list(filter(None, '`~'.join(itemList).split('*-')))
        ]

        for line in lines:

            columns = [
                list(filter(None, l.split('`~')))
                for l in list(filter(None, '`~'.join(line).split('*|')))
            ]

            tr = TableRow()
            table.addElement(tr)

            for input_column in columns:

                column = [i for i in input_column
                          if i != '*:']  #removes all '*:' occurances
                #list(filter((2).__ne__, x))

                number_of_merges = len(input_column) - len(column) + 1

                tc = TableCell(numbercolumnsspanned=number_of_merges)

                tr.addElement(tc)
                para = P(stylename=centerstl)
                n = 0

                while (n < len(column)):

                    item = column[n]

                    ## string
                    stylename, color = '', ''
                    if ("*" in item or "#" in item):
                        n += 1
                        stylename, color = self.splitStyleString(item)
                    data = column[n]
                    if ('\n' in data):
                        dn = 0
                        data_lines = data.split('\n')
                        while (dn < len(data_lines) - 1):
                            txt = Span(
                                text=data_lines[dn],
                                stylename=self.createTextStyle(stylename))
                            para.addElement(txt)
                            tc.addElement(para)
                            para = P(stylename=centerstl)
                            data = data_lines[-1]
                            dn += 1
                    txt = Span(text=data,
                               stylename=self.createTextStyle(stylename))
                    para.addElement(txt)
                    n += 1

                    if color:
                        table_style = self.createTableStyle(color)
                        tc.setAttribute('stylename', color)

                    tc.addElement(para)

        self.doc.text.addElement(table)

    def addImageToDocument(self,
                            image_path,
                            figure_caption,
                            figure_number=1,
                            size_factor=0.4,
                            text_font_size="10pt"):

        pil_image = PIL_Image.open(image_path)
        image_width, image_height = pil_image.size

        frame_width, frame_height = str(round(
            size_factor * (image_width + 2))) + "px", str(
                round(size_factor * (image_height + 2))) + "px"
        img_width, img_height = str(round(
            size_factor * image_width)) + "px", str(
                round(size_factor * image_height)) + "px"

        main_p = P(stylename="Standard")

        mainframestyle = Style(name='mainframestyle',
                               parentstylename="Frame",
                               family="graphic")
        mainframestyle.addElement(
            GraphicProperties(marginleft="0cm",
                              marginright="0cm",
                              margintop="0cm",
                              marginbottom="0cm",
                              wrap="dynamic",
                              numberwrappedparagraphs="no-limit",
                              verticalpos="top",
                              verticalrel="paragraph",
                              horizontalpos="center",
                              horizontalrel="paragraph",
                              padding="0cm",
                              border="none",
                              shadow="none",
                              shadowopacity="100%"))

        self.doc.automaticstyles.addElement(mainframestyle)

        mainframe = Frame(width=frame_width,
                          height=frame_height,
                          anchortype="char",
                          stylename=mainframestyle)

        mainFrameBox = TextBox(minheight="0.041cm", minwidth="0.57cm")

        p = P()

        photoframe = Frame(width=img_width,
                           height=img_height,
                           anchortype="paragraph")
        href = self.doc.addPicture(image_path)
        photoframe.addElement(Image(href=href))
        p.addElement(photoframe)

        figuretextspan = Span(text="Figure {} : ".format(figure_number),
                              stylename=self.createTextStyle('i'))
        p.addElement(figuretextspan)

        figurecaptionspan = Span(text=figure_caption,
                                 stylename=self.createTextStyle(''))
        p.addElement(figurecaptionspan)

        mainFrameBox.addElement(p)

        mainframe.addElement(mainFrameBox)

        main_p.addElement(mainframe)

        self.doc.text.addElement(main_p)

    def addListToDocument(self, itemList, text_font_size='12pt'):

        mixedListSpec = u'1.!\u273f!a)'

        listStyle = easyliststyle.styleFromString('mix1', mixedListSpec, '!',
                                                  '0.8cm',
                                                  easyliststyle.SHOW_ONE_LEVEL)
        self.doc.styles.addElement(listStyle)

        listItem = ListItem()
        listArray = List()

        listArray.setAttribute('stylename', 'mix1')

        para = None

        n = 0

        lines = [
            list(filter(None, l.split('||')))
            for l in list(filter(None, '||'.join(itemList).split('*e')))
        ]

        for line in lines:

            # print(line)

            listItem = ListItem()
            para = P()

            n = 0

            while (n < len(line)):

                item = line[n]

                if ("*" in item):
                    n += 1
                    txt = Span(
                        text=line[n],
                        stylename=self.createTextStyle(
                            item[1:],
                            fontname=self.default_font_face.getAttribute(
                                'name')))
                    para.addElement(txt)
                else:
                    txt = Span(
                        text=line[n],
                        stylename=self.createTextStyle(
                            '',
                            fontname=self.default_font_face.getAttribute(
                                'name')))
                    para.addElement(txt)

                n += 1

            listItem.addElement(para)
            listArray.addElement(listItem)

        self.doc.text.addElement(listArray)

    def addParagraphToDocument(self,
                               text,
                               line_spacing=1.5,
                               paragraph_spacing="0.5cm",
                               text_font_size='12pt'):

        self.justifystl = Style(name='justifystl_{}_{}_{}'.format(
            line_spacing * 100, paragraph_spacing, text_font_size),
                                family="paragraph")
        self.justifystl.addElement(
            ParagraphProperties(textalign="justify",
                                justifysingleword="false",
                                marginbottom=paragraph_spacing,
                                margintop=paragraph_spacing,
                                lineheight="{}%".format(line_spacing * 100),
                                contextualspacing="false"))

        default_stl = copy.deepcopy(
            self.default_text_style.getElementsByType(TextProperties)[0])

        default_stl.setAttribute('fontsize', text_font_size)
        default_stl.setAttribute('fontname',
                                 self.default_font_face.getAttribute('name'))

        self.justifystl.addElement(default_stl)

        self.doc.automaticstyles.addElement(self.justifystl)

        para = P(stylename=self.justifystl)

        txt = Span(text=text)
        para.addElement(txt)

        self.doc.text.addElement(para)

    def addPageBreakToDocument(self):
        withbreak = Style(name="WithBreak",
                          parentstylename="Standard",
                          family="paragraph")
        withbreak.addElement(ParagraphProperties(breakbefore="page"))
        self.doc.automaticstyles.addElement(withbreak)

        p = P(stylename=withbreak, text=u'')
        self.doc.text.addElement(p)


if "__main__" == __name__:

    filepath_and_filename = "document.odt"

    #image
    image_path = 'test.png'
    figure_caption = "Hello"

    #Create Document Object
    document = OdtDocument()

    paragraph = "khuv"#"Lorem ipsum, dolor sit amet consectetur adipisicing elit. Iure unde consequuntur velit obcaecati quas at alias magnam voluptatibus, officia quasi vero aliquid modi molestiae, ad pariatur! Veritatis sunt labore officiis."

    document.addParagraphToDocument("  Justified  ")

    document.addParagraphToDocument(paragraph)

    document.addParagraphToDocument(
        paragraph,
        line_spacing=1.25,
        paragraph_spacing=
        "0.125cm",  # paragraph_spacing <--- this is margin on top and bottom ,total spacing will be double of this
        text_font_size='14pt')

    document.addParagraphToDocument(
        paragraph,
        line_spacing=1.75,
        paragraph_spacing=
        "20pt",  # paragraph_spacing <--- this is margin on top and bottom ,total spacing will be double of this
        text_font_size='16pt')

    # bullet list
    bulletList = [
        '*e', 'A Road ', '*b', 'network ', 'from ', '*i', 'OpenStreetMap', ';',
        '*e', '*b', 'Mode share', ': percentage of commuters ', '*u',
        'cycling to work per', ' administrative region [2011 national census]',
        ';', '*e', 'Reported commuting by bike ', '*bu',
        'origin-destination pairs ', '[2011 national census].'
    ]

    document.addListToDocument(bulletList)

    document.addPageBreakToDocument()  #
    # Image
##    document.addImageToDoucument(image_path,
##                                 figure_caption,
##                                 figure_number=1,
##                                 size_factor=0.42)
    # document.addImageToDoucument(image_path,
    #                              "Hello world",
    #                              figure_number=2,
    #                              size_factor=0.5,
    #                              text_font_size="12pt")

    document.addParagraphToDocument(
        "    ", paragraph_spacing="20pt")  # empty paragrpah as seperator

    # table
    tableString = [
        '*-', '*b', 'Road Type ', '*|', '*b',
        'Super Cyclist \n (<2% population) ', '*|', '*b',
        'Average Cyclist \n (~60% population) ', '*-', 'LTS 1', '*|', '*i',
        '18 kph', '*|', '*i', '18 kph', '*-', 'LTS 1', '*|', '*i', '18 kph',
        '*|', '*i', '18 kph', '*-', 'LTS 1', '*|', '*i', '18 kph', '*|', '*i',
        '18 kph', '*-', 'LTS 1', '*|', '*i', '18 kph', '*|', '*i', '18 kph',
        '*-', '#aaaaff', 'Dismount Condition', '*|', '*bu#ffff00', '18 kph',
        '*:', '*-', '*bi', 'Poor Roads. gravel, stairs, etc.', '*|',
        '*bu#88ff88', '18 kph', '*:',
        '\n(formula obtained from OSRM project)\n size wrapping test string added asdgfknasdgnasgnasdfgn'
    ]

    document.addTableToDocument(tableString)

    document.save(filepath_and_filename)
