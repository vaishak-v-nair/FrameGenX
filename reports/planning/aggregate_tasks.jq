# Restrict this continuation to its explicit logical run. Older main/initial-
# commit reviews must not be mistaken for this plan's tasks.
map(select(.run_id == $run_id and .branch == $branch)
    | .commit as $c | select($commits | index($c) != null))
| group_by(.phase)
| map(. as $entries | (max_by(.run_id).run_id) as $latest
      | $entries | map(select(.run_id == $latest)))
| add
| group_by([.component, (.files | sort), .title])
| map(sort_by(({P1:0,P2:1,P3:2}[.priority] // 99),
              ({"ceo-review":0,"design-review":1,"eng-review":2,"devex-review":3}[.phase] // 99)) | .[0])
| sort_by(({P1:0,P2:1,P3:2}[.priority] // 99),
          ({"ceo-review":0,"design-review":1,"eng-review":2,"devex-review":3}[.phase] // 99))
