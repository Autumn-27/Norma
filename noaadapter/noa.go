package noaadapter

import (
	"github.com/Autumn-27/norma/agentcore"
	"github.com/Autumn-27/norma/noa"
	"github.com/Autumn-27/norma/tool"
)

// Enable wires noa into a set of agentcore Options.
//
// This is the ONLY entry point, and it is all-or-nothing by design. It attaches
// three things at once:
//
//  1. opts.Compactor          — takes over the view (and disables the built-in
//     Compaction, which a session cannot run alongside it)
//  2. opts.Tools += Compress  — the compression tool
//  3. opts.AppendSystemPrompt — the three resident prompt sections
//
// They must live or die together. Wiring them separately makes three ways to
// get it half-right, and each failure is quiet:
//
//   - no Compactor: the model compresses, the panel says it worked, and the
//     view never changes — the compression went nowhere.
//   - no tool: the prompt teaches a tool that does not exist.
//   - no prompt: the model has the tool but not the rules, and writes summaries
//     that drop paths, decisions and error strings. Nothing reports an error;
//     it surfaces hours later as a model that has forgotten what it did.
//
// Not calling Enable is the off switch: nothing is attached, and the built-in
// compaction keeps working exactly as before.
func Enable(opts *agentcore.Options, o Options) error {
	if o.ModelContextLimit == 0 && o.Config == nil {
		o.ModelContextLimit = defaultContextLimit
	}
	if o.OnWarn == nil {
		o.OnWarn = opts.OnWarn
	}
	sess, err := newSession(o)
	if err != nil {
		return err
	}

	opts.Compactor = &Compactor{sess: sess}
	opts.Tools = append(opts.Tools, newCompressTool(sess))
	opts.AppendSystemPrompt = append(opts.AppendSystemPrompt, noa.BuildSystemPrompt(noa.Sections{}))
	// The prompt section is fully static, so it belongs in the cacheable prefix.
	// Left past the boundary it would be re-billed every single request.
	opts.DynamicBoundary = len(opts.SystemPrompt) + len(opts.AppendSystemPrompt)
	return nil
}

// defaultContextLimit is used when the host names no window. It is deliberately
// conservative: guessing high would delay every pressure threshold past the
// point where they could still help.
const defaultContextLimit = 200000

// EnableWithSession is Enable, also returning the session so a host can inspect
// state or archives. The session is owned by the returned wiring; callers must
// not construct a second one over the same directory.
func EnableWithSession(opts *agentcore.Options, o Options) (*Session, error) {
	if err := Enable(opts, o); err != nil {
		return nil, err
	}
	c, ok := opts.Compactor.(*Compactor)
	if !ok {
		return nil, ErrNoArchiveDir
	}
	return c.sess, nil
}

var _ tool.CoreTool = newCompressTool(&Session{cfg: noa.DefaultConfig(1)})
