#!/bin/bash
BASE="http://localhost:8000"

echo "=== A: Java developer (expect: Core Java, Spring, SQL, Verify G+, OPQ32r) ==="
curl -s -X POST $BASE/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I need assessments for a senior Java developer with Spring and SQL"}]}' \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
for r in d['recommendations']:
    print(' -', r['name'])
if not d['recommendations']:
    print(' (no recommendations — reply:', d['reply'][:80], ')')
"

sleep 3

echo ""
echo "=== B: Contact centre (expect: SVAR, Contact Center Simulation) ==="
curl -s -X POST $BASE/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"We are screening 500 entry-level contact centre agents, inbound calls, English US"}]}' \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
for r in d['recommendations']:
    print(' -', r['name'])
if not d['recommendations']:
    print(' (no recommendations — reply:', d['reply'][:80], ')')
"

sleep 3

echo ""
echo "=== C: Vague — expect 0 recs + clarifying question ==="
curl -s -X POST $BASE/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"We need a solution for senior leadership"}]}' \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(' recs:', len(d['recommendations']))
print(' reply:', d['reply'])
"

sleep 3

echo ""
echo "=== D: Legal refusal — expect 0 recs ==="
curl -s -X POST $BASE/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Are we legally required under HIPAA to test all staff who touch patient records?"}]}' \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(' recs:', len(d['recommendations']))
print(' reply:', d['reply'])
"
