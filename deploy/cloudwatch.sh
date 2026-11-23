#!/usr/bin/env bash
# One-time (idempotent) CloudWatch setup: metric filters on the nutribot JSON log + 4 alarms -> SNS email.
# Usage: AWS_REGION=us-east-1 ALERT_EMAIL=you@example.com deploy/cloudwatch.sh
set -euo pipefail
: "${ALERT_EMAIL:?set ALERT_EMAIL}"
REGION="${AWS_REGION:-us-east-1}"
GROUP=nutribot NS=NutriBot
P95_MS="${P95_MS:-40000}"          # baseline avg is ~17 s; tune after a week of data
DAILY_COST_USD="${DAILY_COST_USD:-5}"
REFUSAL_PCT="${REFUSAL_PCT:-30}"
aws() { command aws --region "$REGION" "$@"; }

aws logs create-log-group --log-group-name "$GROUP" 2>/dev/null || true
TOPIC=$(aws sns create-topic --name nutribot-alerts --query TopicArn --output text)
aws sns subscribe --topic-arn "$TOPIC" --protocol email --notification-endpoint "$ALERT_EMAIL" >/dev/null

filter() {  # name, pattern, metric name, value, default
  aws logs put-metric-filter --log-group-name "$GROUP" --filter-name "$1" --filter-pattern "$2" \
    --metric-transformations "metricName=$3,metricNamespace=$NS,metricValue=$4,defaultValue=$5"
}
filter requests  '{ $.request_id = * }'       Requests  1 0
filter errors5xx '{ $.status >= 500 }'        Errors5xx 1 0
filter refusals  '{ $.refused IS TRUE }'      Refusals  1 0
filter latency   '{ $.latency_ms = * }'       LatencyMs '$.latency_ms' 0
filter cost      '{ $.cost_usd = * }'         CostUsd   '$.cost_usd' 0

alarm() {  # name, description, then put-metric-alarm args
  local n=$1 d=$2; shift 2
  aws cloudwatch put-metric-alarm --alarm-name "nutribot-$n" --alarm-description "$d" \
    --alarm-actions "$TOPIC" --treat-missing-data notBreaching "$@"
}
alarm 5xx "3+ failed requests in 5 min" --namespace $NS --metric-name Errors5xx \
  --statistic Sum --period 300 --evaluation-periods 1 --threshold 3 --comparison-operator GreaterThanOrEqualToThreshold
alarm p95-latency "p95 request latency > ${P95_MS} ms for 10 min" --namespace $NS --metric-name LatencyMs \
  --extended-statistic p95 --period 300 --evaluation-periods 2 --threshold "$P95_MS" --comparison-operator GreaterThanThreshold
alarm daily-cost "Spend > \$${DAILY_COST_USD}/day" --namespace $NS --metric-name CostUsd \
  --statistic Sum --period 86400 --evaluation-periods 1 --threshold "$DAILY_COST_USD" --comparison-operator GreaterThanThreshold
alarm refusal-rate "Refusals > ${REFUSAL_PCT}% of requests over 1 h (needs 10+ requests)" \
  --evaluation-periods 1 --threshold "$REFUSAL_PCT" --comparison-operator GreaterThanThreshold \
  --metrics '[{"Id":"pct","Expression":"IF(req>=10, 100*ref/req, 0)","ReturnData":true},
    {"Id":"ref","ReturnData":false,"MetricStat":{"Metric":{"Namespace":"NutriBot","MetricName":"Refusals"},"Period":3600,"Stat":"Sum"}},
    {"Id":"req","ReturnData":false,"MetricStat":{"Metric":{"Namespace":"NutriBot","MetricName":"Requests"},"Period":3600,"Stat":"Sum"}}]'
echo "Done. Confirm the SNS email subscription sent to $ALERT_EMAIL."
