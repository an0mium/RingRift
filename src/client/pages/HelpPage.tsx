import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  TeachingOverlay,
  parseTeachingTopic,
  TEACHING_TOPICS,
} from '../components/TeachingOverlay';
import { StatusBanner } from '../components/ui/StatusBanner';
import { Button } from '../components/ui/Button';
import { ButtonLink } from '../components/ui/ButtonLink';

function formatTopicLabel(topic: string): string {
  return topic
    .split('_')
    .map((word) => (word.length > 0 ? word[0].toUpperCase() + word.slice(1) : word))
    .join(' ');
}

export default function HelpPage() {
  const navigate = useNavigate();
  const { topic } = useParams<{ topic?: string }>();

  const parsedTopic = parseTeachingTopic(topic);
  const hasInvalidTopicParam = !!topic && !parsedTopic;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="container mx-auto px-4 py-10 space-y-6">
        <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="space-y-1">
            <h1 className="text-2xl sm:text-3xl font-bold flex items-center gap-2">
              <img
                src="/ringrift-icon.png"
                alt="RingRift"
                className="w-7 h-7 sm:w-8 sm:h-8 flex-shrink-0"
              />
              <span>Help & Rules</span>
            </h1>
            <p className="text-sm text-slate-400 max-w-2xl">
              Open a topic for a short, practical explanation of the mechanic and what to do next.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <ButtonLink to="/sandbox?preset=learn-basics" size="sm">
              Learn the Basics
            </ButtonLink>
            <ButtonLink to="/sandbox" variant="secondary" size="sm">
              Open Sandbox
            </ButtonLink>
          </div>
        </header>

        {hasInvalidTopicParam ? (
          <StatusBanner
            variant="error"
            title="Unknown help topic"
            actions={
              <Button type="button" variant="secondary" size="sm" onClick={() => navigate('/help')}>
                View topics
              </Button>
            }
          >
            <span className="font-mono text-xs">{topic}</span> is not a supported help topic.
          </StatusBanner>
        ) : null}

        <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {TEACHING_TOPICS.map((t) => (
            <Link
              key={t}
              to={`/help/${t}`}
              className="rounded-xl border border-slate-700 bg-slate-900/60 px-4 py-3 hover:border-emerald-500/60 hover:bg-slate-900 transition-colors"
            >
              <div className="text-sm font-semibold text-slate-100">{formatTopicLabel(t)}</div>
              <div className="text-xs text-slate-400 mt-0.5 font-mono">/help/{t}</div>
            </Link>
          ))}
        </section>

        <TeachingOverlay
          topic={parsedTopic ?? 'ring_placement'}
          isOpen={parsedTopic !== null}
          onClose={() => navigate('/help', { replace: true })}
          position="center"
        />
      </div>
    </div>
  );
}
