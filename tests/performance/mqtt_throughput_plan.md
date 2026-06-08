# MQTT Throughput Plan

Status: `拟开展/未验证`

## Target

- Scenario: mock edge publishers send status/track/alarm events.
- Target throughput: contract target unless measured.
- QoS: QoS1.
- Reliability: message loss, duplicate count, reconnect behavior.

## Procedure

1. Start broker and cloud consumer in an isolated test environment.
2. Publish generated mock events with increasing `deviceId` and `seq`.
3. Record publish rate, broker acknowledgements, cloud accepted count.
4. Inject disconnect windows for 30/60/300s and measure resend behavior.

## Result Table

| Date | Environment | Rate | Loss | Duplicates | Status |
|---|---|---:|---:|---:|---|
| 拟开展/未验证 | 拟开展/未验证 | 拟开展/未验证 | 拟开展/未验证 | 拟开展/未验证 | target |
