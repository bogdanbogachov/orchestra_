# Attention Pooling Analysis

These are pooling-head weights, not transformer self-attention explanations.

- Samples analyzed: 200
- Clean spans found: 186
- Comparable original/noise examples: 184
- Mean attention mass on original query span: 0.8854
- Mean attention mass on injected noise: 0.1141
- Mean per-token weight on original span: 0.083543
- Mean per-token weight on injected noise: 0.019953
- Mean top-5 original-token fraction: 0.8109

## Top-Token Examples

### Example 0

- Text: How do I locate my card? what is the expected timeline?
- Label/pred: 11 / 41
- Top tokens: `?:0.7940:original ?:0.1797:noise Ġcard:0.0182:original Ġtimeline:0.0047:noise Ġwhat:0.0011:noise`

### Example 1

- Text: I still have not received my new card, I ordered over a week ago. what is the expected timeline?
- Label/pred: 11 / 11
- Top tokens: `,:0.5728:original .:0.2907:original ?:0.0973:noise Ġcard:0.0172:original Ġago:0.0066:original`

### Example 2

- Text: Hello there, I ordered a card but it has not arrived. Help please!
- Label/pred: 11 / 11
- Top tokens: `.:0.5735:original Ġarrived:0.4086:original !:0.0094:original Ġnot:0.0073:original Ġbut:0.0007:original`

### Example 3

- Text: Is there a way to know when my card will arrive? is there a formal way to resolve this?
- Label/pred: 11 / 12
- Top tokens: `?:0.5178:original Ġarrive:0.4762:original ?:0.0027:noise Ġthis:0.0016:noise Ġwill:0.0008:original`

### Example 4

- Text: Hello there, My card has not arrived yet.
- Label/pred: 11 / 11
- Top tokens: `Ġyet:0.4377:original .:0.4184:original Ġarrived:0.1436:original Ġcard:0.0001:original Ġnot:0.0001:original`

### Example 5

- Text: When will I get my card? what is the expected timeline?
- Label/pred: 11 / 12
- Top tokens: `?:0.6647:original ?:0.3217:noise Ġtimeline:0.0080:noise Ġcard:0.0045:original Ġexpected:0.0004:noise`

### Example 6

- Text: Do you know if there is a tracking number for the new card you sent me? please advise.
- Label/pred: 11 / 11
- Top tokens: `?:0.8677:original Ġme:0.0520:original Ġsent:0.0369:original .:0.0203:noise Ġcard:0.0148:original`

### Example 7

- Text: I would like to ask, i have not received my card
- Label/pred: 11 / 11
- Top tokens: `Ġcard:0.9235:original Ġmy:0.0242:original Ġreceived:0.0126:original Ġnot:0.0029:original ,:0.0020:noise`

### Example 8

- Text: Good afternoon, still waiting on that card
- Label/pred: 11 / 11
- Top tokens: `Ġcard:0.9922:original Ġwaiting:0.0016:original Ġthat:0.0007:original Ġon:0.0007:original Good:0.0003:noise`

### Example 9

- Text: I’m hoping you can clarify, Is it normal to have to wait over a week for my new card?
- Label/pred: 11 / 11
- Top tokens: `?:0.9464:original Ġcard:0.0533:original Ġweek:0.0002:original Ġnew:0.0001:original Ġfor:0.0000:original`
