import datetime as dt

import streamlit as st

from mlprep import content, plan, progress, srs
from mlprep.content import PHASES
from mlprep.ui import PAGES, data, task_row

d = data()
start = progress.start_date(d)
today = dt.date.today()

st.title("Today")
view = st.date_input("Day to show", value=today, min_value=start,
                     max_value=start + dt.timedelta(days=plan.TOTAL_WEEKS * 7 - 1),
                     help="Look ahead or go back to any day of the plan. Your start date is set on Settings.") \
    if today >= start else today

wd = plan.week_day_for(start, view)
all_tasks = plan.all_tasks()
done_n = sum(progress.is_done(d, t.tid) for t in all_tasks)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Plan progress", f"{done_n / max(1, len(all_tasks)):.0%}", f"{done_n}/{len(all_tasks)} tasks",
          delta_color="off")
deck = plan.unlocked_questions(start, today)
due = [q for q in deck if srs.is_due(d["cards"].get(q.qid), today) and d["cards"].get(q.qid, {}).get("reps", 0) > 0]
new = [q for q in deck if q.qid not in d["cards"]]
c2.metric("Cards due", len(due), f"{len(new)} new", delta_color="off")
days_done = sorted({v for v in d["done"].values()}, reverse=True)
streak, cur = 0, today
while cur.isoformat() in days_done:
    streak += 1
    cur -= dt.timedelta(days=1)
c3.metric("Streak", f"{streak} day{'s' if streak != 1 else ''}")
c4.metric("Days left", max(0, (start + dt.timedelta(days=plan.TOTAL_WEEKS * 7) - today).days))

if today < start:
    st.info(f"Your plan starts on **{start:%A %d %B %Y}**. Change it on Settings, or look at the roadmap.")
    st.page_link(PAGES["settings"], label="Settings", icon=":material/settings:")
    st.stop()
if wd is None:
    st.success("You've reached the end of the 26 weeks. Use the Mock interview and Interview-in-a-week pages "
               "to stay sharp, and keep reviewing cards.")
    st.stop()

week, day = wd
w = content.weeks()[week]
st.subheader(f"Week {week} · Day {day} — {plan.DAY_NAMES[day]}")
st.caption(f"Phase {w.phase} · {PHASES[w.phase][0]}  |  **{w.title}**  |  {view:%A %d %B %Y}")
if day == 1 and w.goal:
    st.info(f"**This week:** {w.goal}")

tasks = plan.tasks_for(week, day)
mins = sum(t.minutes for t in tasks)
left = sum(t.minutes for t in tasks if not progress.is_done(d, t.tid))
st.progress(1 - left / max(1, mins), text=f"About {mins} min planned, {left} min left")
with st.container(border=True):
    for t in tasks:
        task_row(t, "today-")

# catch-up list: anything unfinished before the viewed day
past = []
for i in range((view - start).days):
    pw = plan.week_day_for(start, start + dt.timedelta(days=i))
    if pw:
        past += [t for t in plan.tasks_for(*pw) if not progress.is_done(d, t.tid)]
if past:
    with st.expander(f"Catch-up: {len(past)} unfinished task(s) from earlier days "
                     f"(~{sum(t.minutes for t in past) // 60} h)"):
        st.caption("Tip: if you're more than a week behind, do the exercises and quizzes first; the Review page "
                   "keeps older material alive. Or move your start date on Settings.")
        for t in past[-40:]:
            task_row(t, "catch-")

with st.expander("How a week works"):
    st.markdown(
        """
* **Days 1–5 (Theory + code).** Read one theory section (~25–40 min), do that day's coding exercise
  (implement the stub, then run its tests from the Exercises page), and answer the day's interview questions
  **out loud before** opening the model answer.
* **Day 6 (Lab & project).** The week's interactive lab, plus a mini-project you build in your own repo.
* **Day 7 (Review & mock).** Spaced-repetition review of every due card, a timed mock on this week's
  questions, and a short teach-back note in your own words.
* Plan for about **2 h on weekdays** and **2–3 h on days 6–7**. Questions marked ★ are high-yield.
"""
    )
