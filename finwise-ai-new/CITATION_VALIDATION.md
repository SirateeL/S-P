Investigation of PDF retrieval and citation reliability

Observed data: one DCF PDF, 41 pages, 42 indexed passages. Each of the 42 stored
passages was checked against the normalized text of its original PDF page.
No files were deleted, vectors replaced, or collections migrated.

Code-level causes:
1. A cosine distance >0.55 discarded a passage before generation and could
   discard the whole retrieval result. This uncalibrated cutoff is not an
   evidence-sufficiency test. A regression now covers a relevant distance 0.71.
2. Pure semantic retrieval had no lexical recovery path. Inspection of the
   actual PDF also showed title/contents slides dominating keyword rankings
   for “what is valuation.” Substantive slides need priority over headings.
3. Generic JSON originally did not constrain citations. The previous hardening
   added a schema, but still required the small model to synchronize inline
   [n] labels and an independent integer array. The parser required exact
   agreement and could not verify any supporting quotation.
4. The earlier retry reported the error without including the failed output.

Changes:
- Reciprocal rank fusion combines the top 20 semantic candidates and top 20
  BM25 keyword candidates over stored text. Titles/contents get lower priority.
  Up to eight passages enter the prompt. No fixed distance cutoff claims that
  evidence is insufficient. No embedding-model changes or reindexing required.
- A simpler schema uses paragraphs with text, one source_id, and an exact
  supporting quote. No model-generated page numbers or inline citation labels.
- Backend checks object shapes, integer IDs (rejecting booleans/strings), valid
  page metadata, single-paragraph text, and quotes belonging to the selected
  passage. Whitespace normalization is the only quote normalization allowed.
- Server-rendered inline references use only explicitly selected and validated
  IDs. This does not infer or repair identifiers. Invalid responses are retried
  once with the failed output and reason; exhaustion returns 503 without
  fabricated sources. Empty paragraphs produce a controlled evidence-gap reply.
- The source panel displays the validated supporting quotes as well as the
  retrieved passage and original PDF page link.

Verification: 52 pytest tests passed; 5 unittest checks passed; frontend syntax
and 5 calculator checks passed. The real PDF storage/upload test uses a temp
collection. Compile check also passed. PyMuPDF and urllib3 emit existing
compatibility/deprecation warnings.

Limits: neither 127.0.0.1:5001 nor 127.0.0.1:11434 is reachable from this session.
The exact historic failed model output and live question embedding distances
could not be recovered. These code-level causes and reproductions do not prove
which exact output caused the original error. Live Llama generation and browser
behavior require local verification using the README restart instructions.
Quote membership verifies provenance, not semantic entailment or relevance of
every paraphrased claim. A small model can still misinterpret evidence. Hybrid
retrieval scans stored text per question and is intended for this local library.

Restart and manual testing: see README.md, “Restart and verify an existing
installation.” No GitHub push was performed.
