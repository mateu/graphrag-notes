import sys

with open("crates/agents/src/librarian.rs", "r") as f:
    content = f.read()

search_block = """        // Create the note
        let mut note = Note::new(content)
            .with_type(NoteType::Raw)
            .with_embedding(embedding)
            .with_tags(tags);
        
        if let Some(title) = title {
            note = note.with_title(title);
        }"""

replace_block = """        // Determine title
        let note_title = if let Some(t) = title {
            Some(t)
        } else {
            // Try AI generation
            if let Ok(Some(gen_title)) = self.ml.generate_title(&content).await {
                Some(gen_title)
            } else {
                // Fallback to first line
                content.lines()
                    .find(|l| !l.trim().is_empty())
                    .map(|l| {
                        let l = l.trim();
                        if l.len() > 50 {
                            format!("{}...", &l[..50])
                        } else {
                            l.to_string()
                        }
                    })
            }
        };

        // Create the note
        let mut note = Note::new(content)
            .with_type(NoteType::Raw)
            .with_embedding(embedding)
            .with_tags(tags);
        
        if let Some(t) = note_title {
            note = note.with_title(t);
        }"""

if search_block in content:
    new_content = content.replace(search_block, replace_block)
    with open("crates/agents/src/librarian.rs", "w") as f:
        f.write(new_content)
    print("Successfully updated librarian.rs")
else:
    print("Could not find search block in librarian.rs")
    sys.exit(1)
