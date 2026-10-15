/* htm bound to preact's h: JSX-like templates with no build step. */
import { h } from 'preact';
import htm from 'htm';

export const html = htm.bind(h);
export { h };
