import datetime as dt
import json

import streamlit as st

from mlprep import plan, progress
from mlprep.ui import data

d = data()
st.title("Settings & data")

st.subheader("Plan start date")
st.markdown("Week 1, day 1 falls on this date. Moving it shifts every task; finished tasks stay finished.")
new = st.date_input("Start date", progress.start_date(d))
end = new + dt.timedelta(days=plan.TOTAL_WEEKS * 7 - 1)
st.caption(f"The plan runs to **{end:%A %d %B %Y}** ({plan.TOTAL_WEEKS} weeks).")
if new != progress.start_date(d) and st.button("Save start date", type="primary"):
    d["start_date"] = new.isoformat()
    progress.save(d)
    st.success("Saved.")
    st.rerun()

st.subheader("Where your progress lives")
st.code(str(progress.PATH), language="text")
st.caption("Set the environment variable MLPREP_PROGRESS to keep it elsewhere (e.g. a synced folder). "
           "It's git-ignored so it won't be pushed.")

c1, c2 = st.columns(2)
c1.download_button("Download a backup", json.dumps(d, indent=1), file_name="mlprep-progress.json",
                   mime="application/json")
up = c2.file_uploader("Restore from a backup", type="json")
if up is not None and st.button("Replace my progress with this file"):
    d.clear()
    d.update(json.loads(up.read()))
    progress.save(d)
    st.success("Restored.")
    st.rerun()

with st.expander("Danger zone"):
    if st.button("Reset all progress (tasks, cards, mocks)"):
        st.session_state.confirm_reset = True
    if st.session_state.get("confirm_reset"):
        st.warning("This clears everything except your start date. It can't be undone unless you have a backup.")
        if st.button("Yes, reset", type="primary"):
            keep = d["start_date"]
            d.clear()
            d.update(progress._default())
            d["start_date"] = keep
            progress.save(d)
            st.session_state.confirm_reset = False
            st.rerun()
