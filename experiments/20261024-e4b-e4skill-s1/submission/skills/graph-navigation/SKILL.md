---
name: graph-navigation
description: Procedure for the code-graph tools (search_similar_code, get_code_neighbors, get_code_subgraph). Load it once before your first search to find the code an issue touches and the code that must change with it.
---
# Using the code graph

The graph tools take symbol names, never sentences: a dotted path or suffix such as
`package.module.Class.method`, `Class.method` or `function_name`.

1. **Locate.** For each function, class or method named in the issue (or in its traceback), call
   `search_similar_code` with that name. Open the top hit with `read_file` and a line range around
   the definition, not the whole file.
2. **Expand.** Call `get_code_neighbors` on the function you plan to change. It lists the code linked
   to it by calls, in either direction (callers and callees).
3. **Decide.** From that list, keep the entries in the same package that handle the same data or are a
   parallel path: a sync/async twin, an overload, a sibling subclass, the caller that builds the input.
   Read them; they often need the same fix.
4. **Connect several symbols.** When the issue involves 2–5 symbols, one `get_code_subgraph` call with
   that list shows how they connect.
5. **Budget.** About 4 graph calls per task is enough. If a call returns nothing or an error, switch to
   `run_command` with `grep -rn "name" --include=*.py /workspace` instead of retrying it.
