# Bible-JSON

The Bible in JSON format.

## Structure

This is what it would look like if you were to open up `./JSON/Psalms/3.json`.

```json
{
  "book_name": "Psalms",
  "chapter": 3,
  "verses": [
    {
      "book_name": "Psalms",
      "chapter": 3,
      "verse": 1,
      "text": "LORD, how are they increased that trouble me! many <i>are</i> they that rise up against me.",
      "header": "¶ A Psalm of David, when he fled from Absalom his son.",
      "footer": ""
    },
    ...
  ]
}
```

`header` is a string of additional information about the chapter, such as "A Psalm of David" above a Psalm.

`footer` is a string additional information put at the end of a chapter.

`verses` is an object array. Inside `verses` are objects with these:

- `book_id` is a 3-letter string abbreviation of the name of the chapter's book.
- `book_name` is a sting containing the name of the book of which this chapter is from.
- `chapter` is an integer which is the chapter number.
- `verse` is an integer which is the verse number.
- `text` is a string containing the text of the verse. You will find `<i>` and `</i>` around italicized words.
- `info` is a string that would go above the verse. This only applies to Psalm 119.
